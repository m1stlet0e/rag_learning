from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder, HumanMessagePromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
import torch

# 改进的向量检索
def load_retriever():
    return FAISS.load_local(
        'LLM.faiss',
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'mps'},
            encode_kwargs={'normalize_embeddings': True}
        ),
        allow_dangerous_deserialization=True
    ).as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.6}
    )


# 优化模型加载
def load_model():
    model_id = "Qwen/Qwen-1_8B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        pad_token='<|endoftext|>'
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True  # 新增内存优化
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# 改进的Prompt模板
def create_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一个专业助理，请严格根据上下文回答问题。如果上下文不相关，直接回答不知道。"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate.from_template(
            "相关上下文：\n{context}\n\n"
            "请根据以上上下文回答：\n{query}"
        )
    ])


# 智能检索判断逻辑
def should_retrieve(query):
    # 过滤通用问候/不需要上下文的查询
    generic_phrases = ["你好", "hi", "hello", "您好"]
    return not any(p in query.lower() for p in generic_phrases)


# 主流程
def main():
    retriever = load_retriever()
    llm_pipeline = HuggingFacePipeline(pipeline=load_model())
    prompt_template = create_prompt()

    chat_chain = (
        {
            "context": lambda x: "\n".join([d.page_content for d in retriever.invoke(x["query"])])
                          if should_retrieve(x["query"]) else "无相关上下文",
            "query": itemgetter("query"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt_template
        | llm_pipeline
        | StrOutputParser()
    )

    chat_history = []
    while True:
        try:
            query = input("\n用户：").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                break

            response = chat_chain.invoke({
                "query": query,
                "chat_history": chat_history
            })

            # 优化后的清洗函数
            def clean_text(text):
                markers = ["<|endoftext|>", "助手：", "Assistant：", "System:"]
                for m in markers:
                    text = text.replace(m, "")
                last_punct = max(text.rfind('。'), text.rfind('！'), text.rfind('？'))
                return text[:last_punct+1] if last_punct != -1 else text.strip()

            clean_response = clean_text(response)
            print(f"\n助手：{clean_response}")

            # 维护历史记录（保留消息对象）
            chat_history.extend([
                HumanMessage(content=query),
                AIMessage(content=clean_response)
            ])
            chat_history = chat_history[-4:]  # 保留最近2轮对话

        except Exception as e:
            print(f"出错：{str(e)}")
            chat_history = []

if __name__ == "__main__":
    main()