
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from retry import retry



def get_chatgpt_field(title, abstract=None, sys_content=None, usr_prompt=None, extra_prompt=True):
    if not sys_content:
        sys_content = (
            "You are a profound researcher who is good at identifying the topic keyword from paper's title and "
            "abstract. The keyword will be used to retrieve related paper from online scholar search engines."
        )
    if not usr_prompt:
        usr_prompt = (
            "Identifying the topic of the paper based on the given title and abstract. So that I can use it as "
            "keyword to search highly related papers from Google Scholar. Avoid using broad or overly general "
            "term like 'deep learning', 'taxonomy', or 'surveys'. Instead, focus on keyword that are unique "
            "and directly pertinent to the paper's subject. Answer with the word only in the "
            "following format: xxx"
        )

    messages = [SystemMessage(content=sys_content)]

    if abstract and extra_prompt:
        extra_abs_content = '''
        Given Title: A Survey of Self-Supervised and Few-Shot Object Detection
        Given Abstract: Labeling data is often expensive and time-consuming, especially for tasks such as object detection and instance segmentation, which require dense labeling of the image. While few-shot object detection is about training a model on novel(unseen) object classes with little data, it still requires prior training on many labeled examples of base(seen) classes. On the other hand, self-supervised methods aim at learning representations from unlabeled data which transfer well to downstream tasks such as object detection. Combining few-shot and self-supervised object detection is a promising research direction. In this survey, we review and characterize the most recent approaches on few-shot and self-supervised object detection. Then, we give our main takeaways and discuss future research directions. Project page: https://gabrielhuang.github.io/fsod-survey/
        '''
        messages.append(HumanMessage(content=f"{usr_prompt}\n\n{extra_abs_content}"))
        messages.append(AIMessage(content='few-shot object detection'))

    content = f"{usr_prompt}\nGiven Title: {title}\n"
    if abstract:
        content += f"Given Abstract: {abstract}"
    messages.append(HumanMessage(content=content))

    chat = ChatOpenAI(model="gpt-3.5-turbo")
    return chat.invoke(messages).content


@retry()
def get_chatgpt_fields(title, abstract, extra_prompt=True, sys_content=None, usr_prompt=None):
    if not sys_content:
        sys_content = (
            "You are a profound researcher who is good at conducting a literature review based on the given title and abstract."
        )
    if not usr_prompt:
        usr_prompt = (
            "Given title and abstract, please provide 5 searching keywords for me so that I can use them as "
            "keywords to search highly related papers from Google Scholar or Semantic Scholar. Please avoid "
            "responding with overly general keywords such as deep learning, taxonomy, or surveys, etc., "
            "and provide the output in descending order of relevance to the keywords. Answer with the words "
            "only in the following format: xxx,xxx,xxx"
        )

    messages = [SystemMessage(content=sys_content)]

    if extra_prompt:
        messages.append(HumanMessage(content=f"{usr_prompt}\n Given Title: Diffusion Models in Vision: A Survey \nGiven Abstract: Denoising diffusion models represent a recent emerging topic in computer vision..."))
        messages.append(AIMessage(content='Denoising diffusion models,deep generative modeling,diffusion models,image generation,noise conditioned score networks'))

    messages.append(HumanMessage(content=f"{usr_prompt}\nGiven Title: {title}\nGiven Abstract: {abstract}"))

    chat = ChatOpenAI(model="gpt-3.5-turbo")
    return chat.invoke(messages).content
