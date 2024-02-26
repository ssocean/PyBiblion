#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author      : YongJie-Xie
@Contact     : fsswxyj@qq.com
@DateTime    : 2022-09-18 18:07
@Description : 支持订阅更新、节点切换、节点检测功能的 Clash 代理工具
@FileName    : clash_proxy
@License     : MIT License
@ProjectName : MugwortTools
@Software    : PyCharm
@Version     : 1.0
"""
import io
import json
import os.path
import random
import re
import socket
import sys
import zipfile
from subprocess import Popen, PIPE
from threading import Thread
from typing import Any, Callable, Dict, List, NoReturn, Optional

from mugwort import Logger # pip install mugwort[all]

try:
    import requests.adapters
    import yaml
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.schedulers.blocking import BlockingScheduler
except ImportError:
    raise ImportError(
        'Tool `proxy.clash` cannot be imported.',
        'Please execute `pip install mugwort[proxy-clash]` to install dependencies first.'
    )

__all__ = [
    'ClashConfig',
    'ClashProxy',
]

requests.adapters.DEFAULT_RETRIES = 7


class ClashConfig:
    """配置类，用于配置代理程序和观察者"""

    _default_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0',
    }

    def __init__(
            self, workdir: str = None, logger: Logger = None,
            *,
            # 代理程序相关配置
            subscribe_link: str = None,
            subscribe_include_keywords: List[str] = None, subscribe_exclude_keywords: List[str] = None,
            listen_host: str = '127.0.0.1', listen_port: int = 0,
            manage_host: str = '127.0.0.1', manage_port: int = 0,
            # 观察者相关配置
            watcher_enable: bool = True, watcher_blocking: bool = False,
            watcher_job_updater_enable: bool = True, watcher_job_updater_config: dict = None,  #
            watcher_job_changer_enable: bool = True, watcher_job_changer_config: dict = None,  #
            watcher_job_checker_enable: bool = True, watcher_job_checker_config: dict = None,  #
            # 观察者【节点检测】功能的请求参数和校验函数
            watcher_job_checker_requests_kwargs: Dict[str, Any] = None,
            watcher_job_checker_verify_function: Callable = None,
            dump_yaml_pth = None
    ):
        """
        :param workdir: 工作目录，默认为当前用户的家目录
        :param subscribe_link: 订阅链接，为空时需手动放置代理配置文件到工作目录
        :param subscribe_include_keywords: 更新订阅时包含的关键词列表
        :param subscribe_exclude_keywords: 更新订阅时排除的关键词列表
        :param listen_host: 混合代理监听主机，默认为 127.0.0.1
        :param listen_port: 混合代理监听端口，默认为 8000-9000 的随机端口
        :param manage_host: 外部管理监听主机，默认为 127.0.0.1
        :param manage_port: 外部管理监听端口，默认为 8000-9000 的随机端口
        :param watcher_enable: 观察者开关
        :param watcher_blocking: 观察者是否阻塞主线程
        :param watcher_job_updater_enable: 观察者【订阅更新】功能开关
        :param watcher_job_updater_config: 观察者【订阅更新】功能的调度参数，具体参见 apscheduler 使用文档
        :param watcher_job_changer_enable: 观察者【节点切换】功能开关
        :param watcher_job_changer_config: 观察者【节点切换】功能的调度参数，具体参见 apscheduler 使用文档
        :param watcher_job_checker_enable: 观察者【节点检测】功能开关
        :param watcher_job_checker_config: 观察者【节点检测】功能的调度参数，具体参见 apscheduler 使用文档
        :param watcher_job_checker_requests_kwargs: 观察者【节点检测】功能的请求参数，即传入 requests 请求函数的参数
        :param watcher_job_checker_verify_function: 观察者【节点检测】功能的校验函数，回调函数传参为 Response 对象
        """
        self._logger = logger or Logger('Clash')
        self.dump_yaml_pth = dump_yaml_pth
        # 初始化工作目录
        if workdir is None:
            workdir = os.path.join(os.path.expanduser('~'), '.clash')
        elif workdir == '.':
            cwd = os.path.dirname(sys.executable if hasattr(sys, 'frozen') else sys.modules['__main__'].__file__)
            workdir = os.path.join(cwd, '.clash')
        elif os.path.isdir(workdir) is False:
            raise ValueError('当前 Clash 工作目录不可用')
        self._workdir = os.path.abspath(workdir)
        os.makedirs(self._workdir, exist_ok=True)
        self._logger.info('[ClashConfig] 当前 Clash 工作目录：%s', self._workdir)

        # 初始化订阅信息
        self._subscribe_link = subscribe_link
        self._subscribe_include_regex = '|'.join(x.replace('|', r'\|') for x in subscribe_include_keywords or [])
        self._subscribe_exclude_regex = '|'.join(x.replace('|', r'\|') for x in subscribe_exclude_keywords or [])
        self._logger.info('[ClashConfig] 订阅链接：%s', self._subscribe_link)
        self._logger.info('[ClashConfig] 节点匹配正则规则：%s', self._subscribe_include_regex)
        self._logger.info('[ClashConfig] 节点排除正则规则：%s', self._subscribe_exclude_regex)

        # 初始化监听地址
        self._listen_host = listen_host
        self._listen_port = self._random_unused_port(8000, 9000, listen_host) if listen_port == 0 else listen_port
        self._logger.info('[ClashConfig] 混合代理接口：%s', self.get_listen_address())
        self._logger.info('请在系统代理中设置该接口: %s', self.get_listen_address())
        # 初始化管理地址
        self._manage_host = manage_host
        self._manage_port = self._random_unused_port(8000, 9000, manage_host) if manage_port == 0 else manage_port
        self._logger.info('[ClashConfig] 外部管理接口：%s', self.get_manage_address())

        # 初始化观察者
        self._watcher_enable = watcher_enable
        if self._watcher_enable:
            self._watcher_blocking = watcher_blocking

        if self._watcher_enable:
            # 初始化观察者【订阅更新】功能
            self._watcher_job_updater_enable = watcher_job_updater_enable
            if self._watcher_job_updater_enable and self._subscribe_link is None:
                self._watcher_job_updater_enable = False
                self._logger.warning('[ClashConfig] 未配置订阅链接，已关闭观察者【订阅更新】功能')
            if self._watcher_job_updater_enable:
                # 调度参数
                if watcher_job_updater_config is None:
                    watcher_job_updater_config = {'trigger': 'cron', 'hour': 2}
                self._watcher_job_updater_config = watcher_job_updater_config
                self._logger.info(
                    '[ClashConfig] 观察者【订阅更新】功能已启用\n调度参数：%s',
                    self._watcher_job_updater_config
                )

            # 初始化观察者【节点切换】功能
            self._watcher_job_changer_enable = watcher_job_changer_enable
            if self._watcher_job_changer_enable:
                # 调度参数
                if watcher_job_changer_config is None:
                    watcher_job_changer_config = {'trigger': 'interval', 'hours': 1}
                self._watcher_job_changer_config = watcher_job_changer_config
                self._logger.info(
                    '[ClashConfig] 观察者【节点切换】功能已启用\n调度参数：%s',
                    self._watcher_job_changer_config
                )

            # 初始化观察者【节点检测】功能
            self._watcher_job_checker_enable = watcher_job_checker_enable
            if self._watcher_job_checker_enable:
                # 调度参数
                if watcher_job_checker_config is None:
                    watcher_job_checker_config = {'trigger': 'interval', 'seconds': 30}
                self._watcher_job_checker_config = watcher_job_checker_config
                # 配置观察者【节点检测】功能的请求参数
                if watcher_job_checker_requests_kwargs is None:
                    watcher_job_checker_requests_kwargs = {
                        'method': 'HEAD', 'url': 'https://www.google.com', 'headers': {'User-Agent': 'curl/7.83.1'},
                    }
                self._watcher_job_checker_requests_kwargs = {
                    k.lower(): v for k, v in watcher_job_checker_requests_kwargs.items()
                }
                # 配置观察者【节点检测】功能的校验函数
                if watcher_job_checker_verify_function is None:
                    def watcher_job_checker_verify_function(response: requests.Response):
                        return response.status_code == 200
                self._watcher_job_checker_verify_function = watcher_job_checker_verify_function
                self._logger.info(
                    '[ClashConfig] 观察者【节点检测】功能已启用\n调度参数：%s\n请求参数：%s',
                    self._watcher_job_checker_config,
                    json.dumps(watcher_job_checker_requests_kwargs, indent=2, ensure_ascii=False),
                )

        # 初始化代理程序
        if os.path.exists(self.get_executor_filepath()) is False:
            self._download_executor(self._workdir)

    def get_executor_filepath(self) -> str:
        """获取执行文件路径"""
        return os.path.join(self._workdir, 'clash-windows-amd64.exe')

    def get_config_filepath(self) -> str:
        """获取配置文件路径"""
        return os.path.join(self._workdir, 'config.yaml')

    def get_launch_command(self) -> str:
        """获取启动命令"""
        return '%s -d %s' % (self.get_executor_filepath(), self._workdir)

    def get_listen_address(self, schema: str = 'http') -> str:
        """获取混合代理监听地址"""
        listen_host = '127.0.0.1' if self._listen_host == '0.0.0.0' else self._listen_host
        listen_port = self._listen_port
        return '%s://%s:%s' % (schema, listen_host, listen_port)

    def get_manage_address(self, schema: str = 'http') -> str:
        """获取外部管理监听地址"""
        manage_host = '127.0.0.1' if self._manage_host == '0.0.0.0' else self._manage_host
        manage_port = self._manage_port
        return '%s://%s:%s' % (schema, manage_host, manage_port)

    def update_subscribe(self) -> NoReturn:
        """更新订阅功能的下载、解析和保存部分"""
        if self._subscribe_link is None:
            return
        if self.dump_yaml_pth:
            self._logger.info(f'[ClashConfig] 正在读取yaml文件:{self.dump_yaml_pth}')
            with open(self.dump_yaml_pth, "r",encoding='utf-8') as file:
                subscribe_yaml = yaml.safe_load(file)
        else:
            self._logger.info('[ClashConfig] 正在更新订阅信息')
            response = requests.get(self._subscribe_link, headers=self._default_headers, timeout=7)
            subscribe_yaml = yaml.safe_load(response.content)
        # self._logger.info('[ClashConfig] 流量信息：%s', response.headers.get('Subscription-Userinfo'))
        self._logger.info('[ClashConfig] 节点总数：%d', len(subscribe_yaml['proxies']))

        subscribe_config = {
            'mode': 'global', 'log-level': 'warning',
            'bind-address': self._listen_host, 'mixed-port': self._listen_port,
            'external-controller': '%s:%s' % (self._manage_host, self._manage_port),
            'proxies': subscribe_yaml['proxies'],
        }

        if self._subscribe_include_regex:
            subscribe_config['proxies'] = [
                proxy for proxy in subscribe_config['proxies']
                if re.findall(self._subscribe_include_regex, proxy['name'])
            ]

        if self._subscribe_exclude_regex:
            subscribe_config['proxies'] = [
                proxy for proxy in subscribe_config['proxies']
                if not re.findall(self._subscribe_exclude_regex, proxy['name'])
            ]

        if self._subscribe_include_regex or self._subscribe_exclude_regex:
            self._logger.info('[ClashConfig] 过滤后节点数量：%d', len(subscribe_config['proxies']))

        config_filepath = self.get_config_filepath()
        with open(config_filepath, 'w', encoding='utf8') as file:
            yaml.dump(subscribe_config, file, allow_unicode=True)
        self._logger.info('[ClashConfig] 当前 Clash 配置文件保存位置：%s', config_filepath)

    @classmethod
    def _random_unused_port(cls, start: int, stop: int, host: str = '127.0.0.1') -> int:
        """获取未使用的随机端口"""
        port = random.randint(start, stop)
        while cls.check_port_occupied(host, port, times=1) is True:
            port = random.randint(start, stop)
        return port

    @staticmethod
    def check_port_occupied(host: str, port: int, times: int = 3) -> bool:
        """检测端口的占用情况"""
        host = '127.0.0.1' if host == '0.0.0.0' else host
        while times > 0:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                if sock.connect_ex((host, port)) == 0:
                    return True
            except OSError:
                pass
            finally:
                times -= 1
        return False

    def _download_executor(self, folder: str) -> NoReturn:
        """下载 Clash 代理程序"""
        zfs = [file for file in os.listdir(folder) if re.match(r'clash-windows-amd64-v\d+.\d+.\d+.zip', file)]
        if zfs:
            self._logger.info('[ClashConfig] 已检测到 Clash 代理程序压缩包，尝试从本地解压')
            if len(zfs) > 1:
                try:
                    from packaging.version import Version
                except ImportError:
                    from distutils.version import StrictVersion as Version
                zfs = list(sorted(zfs, key=lambda s: Version(s[21:-4])))
            self._logger.info('[ClashConfig] 解压 Clash 代理程序压缩包：%s', zfs[-1])
            with zipfile.ZipFile(os.path.join(folder, zfs[-1]), 'r') as zf:
                zf.extractall(folder)
        else:
            self._logger.info('[ClashConfig] 未检测到 Clash 代理程序，尝试从 Github 下载')

            # 获取最新版本号
            url = 'https://api.github.com/repos/Dreamacro/clash/releases/latest'
            response_json = requests.get(url, timeout=7).json()
            tag = response_json['tag_name']
            self._logger.info('[ClashConfig] 当前 Clash 最新版本：%s', tag)

            # 下载最新版本执行程序
            url = 'https://github.com/Dreamacro/clash/releases/download/%s/clash-windows-amd64-%s.zip' % (tag, tag)
            self._logger.info('[ClashConfig] 正在下载代理程序，耗时过多请手动下载\n下载地址：%s\n存放目录：%s', url,
                              folder)
            response = requests.get(url)
            self._logger.info('[ClashConfig] 代理程序下载完毕，正在解压')
            buffer = io.BytesIO()
            buffer.write(response.content)
            with zipfile.ZipFile(buffer, 'r') as zf:
                zf.extractall(folder)
        self._logger.info('[ClashConfig] 当前 Clash 代理程序保存位置：%s', folder)

    @property
    def workdir(self) -> str:
        return self._workdir

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def listen_host(self) -> str:
        return self._listen_host

    @property
    def listen_port(self) -> int:
        return self._listen_port

    @property
    def manage_host(self) -> str:
        return self._manage_host

    @property
    def manage_port(self) -> int:
        return self._manage_port

    @property
    def watcher_enable(self) -> bool:
        return self._watcher_enable

    @property
    def watcher_blocking(self) -> bool:
        return self._watcher_blocking

    @property
    def watcher_job_updater_enable(self) -> bool:
        return self._watcher_job_updater_enable

    @property
    def watcher_job_updater_config(self) -> dict:
        return self._watcher_job_updater_config

    @property
    def watcher_job_changer_enable(self) -> bool:
        return self._watcher_job_changer_enable

    @property
    def watcher_job_changer_config(self) -> dict:
        return self._watcher_job_changer_config

    @property
    def watcher_job_checker_enable(self) -> bool:
        return self._watcher_job_checker_enable

    @property
    def watcher_job_checker_config(self) -> dict:
        return self._watcher_job_checker_config

    @property
    def watcher_job_checker_requests_kwargs(self) -> Dict[str, Any]:
        return self._watcher_job_checker_requests_kwargs

    @property
    def watcher_job_checker_verify_function(self) -> Callable:
        return self._watcher_job_checker_verify_function


class _ClashManager:
    """管理类，用于管理代理程序"""

    def __init__(self, config: ClashConfig):
        self._config = config
        self._logger = self._config.logger

    def reload_subscribe(self) -> bool:
        """请求代理程序重新加载配置文件"""
        url = '%s/configs' % self._config.get_manage_address()
        try:
            response = requests.put(url, data='{}', timeout=7)
            return response.status_code == 204
        except requests.RequestException:
            pass
        return False

    def update_subscribe(self, reload: bool = True) -> bool:
        """更新订阅信息并请求代理程序重新加载配置文件"""
        try:
            self._config.update_subscribe()
            if os.path.exists(self._config.get_config_filepath()) is False:
                raise RuntimeError('配置文件不存在')
            if reload and self.reload_subscribe() is False:
                raise RuntimeError('重新加载配置文件失败')
            return True
        except RuntimeError as e:
            self._logger.error('[ClashManager] 更新订阅失败：%s', e)
        except Exception as e:
            self._logger.exception(e)
        return False

    def get_proxy_nodes(self) -> list:
        """获取全部代理节点"""
        url = '%s/proxies' % self._config.get_manage_address()
        try:
            response_json = requests.get(url, timeout=7).json()
            proxies = [
                proxy['name'] for proxy in response_json['proxies'].values() if 'Shadowsocks' in proxy['type']
            ]
            return proxies
        except requests.RequestException:
            pass
        return []

    def change_proxy_node(self, proxy_name: str) -> bool:
        """请求代理程序切换代理节点"""
        url = '%s/proxies/GLOBAL' % self._config.get_manage_address()
        data = json.dumps({'name': proxy_name}).encode()
        try:
            response = requests.put(url, data, timeout=7)
            if response.status_code == 204:
                proxy_ip = self.get_proxy_ip()
                if proxy_ip:
                    self._logger.info('节点ip：%s', proxy_ip)
                    return True
        except requests.RequestException:
            pass
        self._logger.warning('[ClashManager] 切换代理失败')
        return False

    def change_proxy_node_random(self) -> bool:
        """获取全部代理节点后随机选取并请求代理程序切换至该代理节点"""
        proxies = self.get_proxy_nodes()
        if not proxies:
            self._logger.warning('[ClashManager] 无可选代理')
            return False

        proxy = random.choice(proxies)
        self._logger.info('切换节点：%s', proxy)

        return self.change_proxy_node(proxy)

    def get_proxy_ip(self) -> Optional[str]:
        """获取代理节点的网络出口地址"""
        proxies = {'all': self._config.get_listen_address()}
        urls = ['http://ipinfo.io/ip', 'http://ifconfig.me', 'http://api.ipify.org', 'http://ifconfig.me']  # noqa
        for url in urls:
            try:
                response = requests.get(url, headers={'User-Agent': 'curl/7.54'}, timeout=7, proxies=proxies)
                if response.status_code == 200:
                    return response.text
            except requests.RequestException:
                continue
        return None


class _ClashWatcher:
    """观察者，用于监视和管理代理程序"""

    def __init__(self, config: ClashConfig):
        self._config = config
        self._logger = self._config.logger
        self._manager = _ClashManager(config)

        if self._config.watcher_blocking:
            self._scheduler = BlockingScheduler(timezone='Asia/Shanghai')
        else:
            self._scheduler = BackgroundScheduler(timezone='Asia/Shanghai')

        if self._config.watcher_job_updater_enable:
            self._scheduler.add_job(self.job_updater_subscribe, **self._config.watcher_job_updater_config)
        if self._config.watcher_job_changer_enable:
            self._scheduler.add_job(self.job_changer_proxy_node, **self._config.watcher_job_changer_config)
        if self._config.watcher_job_checker_enable:
            self._scheduler.add_job(self.job_checker_proxy_node, **self._config.watcher_job_checker_config)

    def job_updater_subscribe(self):
        self._logger.info('[ClashWatcher] <定时更新订阅>')
        self._manager.update_subscribe()

    def job_changer_proxy_node(self):
        self._logger.info(f'定时切换节点启动')
        self._manager.change_proxy_node_random()

    def job_checker_proxy_node(self, times: int = 3):
        self._logger.info('[ClashWatcher] <定时检测节点>')
        requests_kwargs = self._config.watcher_job_checker_requests_kwargs.copy()
        requests_kwargs.update({'timeout': 7, 'proxies': {'all': self._config.get_listen_address()}})
        verify_function = self._config.watcher_job_checker_verify_function
        while times > 0:
            try:
                response = requests.request(**requests_kwargs)
                if verify_function(response) is False:
                    raise RuntimeError('代理失败')
            except (requests.RequestException, RuntimeError):
                continue
            else:
                proxy_ip = self._manager.get_proxy_ip()
                if proxy_ip:
                    self._logger.info('[ClashWatcher] 代理程序运行正常，当前出口：%s', proxy_ip)
                    break
            finally:
                times -= 1
        else:
            self._logger.warning('[ClashWatcher] 代理程序运行异常，切换代理节点')
            self._manager.change_proxy_node_random()

    def startup(self):
        try:
            self._scheduler.start()
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        if self._scheduler.running:
            self._scheduler.shutdown()

    def __del__(self):
        if hasattr(self, '_scheduler'):
            self.shutdown()


class ClashProxy:
    # 代理程序的进程
    _process = None

    def __init__(self, config: ClashConfig):
        try:
            self._config = config
            self._logger = self._config.logger
            self._manager = _ClashManager(config)
            self._manager.update_subscribe(reload=False)
            if self._config.watcher_enable:
                self._watcher = _ClashWatcher(config)
        except Exception as e:
            self._logger.critical('[Clash] 代理程序初始化异常')
            self._logger.exception(e)
            sys.exit(1)

    def startup(self) -> NoReturn:
        self._logger.info('[Clash] 启动代理程序')
        self._process = Popen(self._config.get_launch_command(), stdout=PIPE, bufsize=-1)

        self._logger.info('[Clash] 捕获代理程序日志')
        Thread(target=self._log_listener).start()

        self._logger.info('[Clash] 等待代理程序响应')
        host, port = self._config.manage_host, self._config.manage_port
        while self._config.check_port_occupied(host, port, times=10) is False:
            self._logger.warning('代理程序仍未响应，继续等待')

        self._logger.info('[Clash] 切换代理节点')
        self._manager.reload_subscribe()
        self._manager.change_proxy_node_random()

        if self._config.watcher_enable:
            self._logger.info('[Clash] 启动代理程序观察者')
            self._watcher.startup()

    def shutdown(self) -> NoReturn:
        if self._config.watcher_enable:
            self._watcher.shutdown()
        if self._process:
            self._process.kill()
            while self._process.poll() is None:
                pass
            self._process = None

    def get_proxy_address(self, schema: str = 'http') -> str:
        """获取混合代理监听地址，支持 HTTP 和 SOCKS 协议"""
        return self._config.get_listen_address(schema)

    def _log_listener(self) -> NoReturn:
        """捕获代理程序的日志"""
        for line in iter(io.TextIOWrapper(self._process.stdout)):
            if 'Only one usage of each socket address' in line:
                continue
            if line.startswith('time='):
                level, msg = line[39:-2].split(' ', 1)
                getattr(self._logger, level)('[Clash] 代理程序日志：%s', msg[5:])
            else:
                self._logger.warning(line.strip())
            if 'MMDB' in line:
                url = 'https://cdn.jsdelivr.net/gh/Dreamacro/maxmind-geoip@release/Country.mmdb'
                self._logger.info('[Clash] 正在下载 MMDB 文件\n下载地址：%s\n保存位置：%s', url, self._config.workdir)

    @property
    def config(self):
        return self._config

    @property
    def manager(self):
        return self._manager
