from clash_proxy import ClashProxy, ClashConfig
#,'澳大利亚','印度','韩国','加拿大','德国','台湾','日本','新加坡'
# pip install mugwort[proxy-clash]
ClashProxy(ClashConfig(
    listen_port=7890,
    subscribe_link='',
    dump_yaml_pth='clash.yaml',
    # subscribe_include_keywords=['美国','香港','日本','台湾'],
    subscribe_exclude_keywords=['过期时间', '剩余流量', '官网'],
    watcher_blocking=True,
    # 默认每天凌晨两点更新订阅
    watcher_job_updater_enable=True,
    watcher_job_updater_config={'trigger': 'cron', 'hour': 12},
    # 默认每间隔一小时切换节点
    watcher_job_changer_enable=True,
    watcher_job_changer_config={'trigger': 'interval', 'seconds': 6},
    # 默认每间隔三十秒检测节点
    watcher_job_checker_enable=True,
    watcher_job_checker_config={'trigger': 'interval', 'seconds': 60*60*48},
)).startup()