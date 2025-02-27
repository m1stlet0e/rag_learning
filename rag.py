from operator import itemgetter
import torch
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ========== 配置区 ==========
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"  # 更优的中文Embedding模型
LLM_MODEL = "Qwen/Qwen-1_8B-Chat"  # 确保模型路径正确
FAISS_INDEX_PATH = "LLM.faiss"  # 向量数据库路径
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # 自动检测设备

# ========== 初始化Embedding模型 ==========
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)

# ========== 加载向量数据库 ==========
try:
    vector_db = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "score_threshold": 0.65}
    )
except Exception as e:
    print(f"加载向量数据库失败: {str(e)}")
    exit()

# ========== 初始化大语言模型 ==========
try:
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL,
        trust_remote_code=True,
        pad_token='<|endoftext|>'  # 重要：设置填充token
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map=DEVICE,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id  # 使用定义的pad token
    )

    chat = HuggingFacePipeline(pipeline=pipe)
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    exit()

# ========== 提示词模板 ==========
system_template = """你是一个专业的人工智能助手，请严格遵守以下规则：
1. 仅使用提供的上下文回答问题 
2. 如果上下文不相关，回答"我不知道"
3. 使用简洁的中文回答"""
system_prompt = SystemMessagePromptTemplate.from_template(system_template)

user_template = """上下文：
{context}

问题：{query}

请基于上下文回答："""
user_prompt = HumanMessagePromptTemplate.from_template(user_template)

full_chat_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    MessagesPlaceholder(variable_name="chat_history"),
    user_prompt
])

# ========== 对话链 ==========
chat_chain = {
                 "context": itemgetter("query") | retriever | (
                     lambda docs: "\n\n".join([d.page_content for d in docs])),
                 "query": itemgetter("query"),
                 "chat_history": itemgetter("chat_history")
             } | full_chat_prompt | chat

# ========== 对话循环 ==========
chat_history = []
while True:
    try:
        query = input('\n用户提问: ').strip()
        if not query:
            continue

            # 执行对话链
        response = chat_chain.invoke({
            "query": query,
            "chat_history": chat_history
        })

        # 提取回答内容
        answer = response.content.split("回答：")[-1].strip()  # 提取最终回答

        # 显示处理结果
        print(f"\n助手回答: {answer}")

        # 维护对话历史
        chat_history.extend([
            HumanMessage(content=query),
            response
        ])
        chat_history = chat_history[-4:]  # 保留最近2轮对话

    except KeyboardInterrupt:
        print("\n对话结束")
        break

    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        chat_history = []  # 重置对话历史
