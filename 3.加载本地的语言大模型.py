#加载模型
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from transformers import AutoTokenizer, AutoModelForCausalLM,TextIteratorStreamer
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple
import torch
import os
from pathlib import Path


class deepseek_local_llm(LLM):
    max_tokens: int = 1000
    do_sample: bool = True  # 生成文本时是否采用采样策略
    temperature: float = 0.7  # 温度
    model_dir: Optional[str] = None
    model_name: Optional[str] = None
    top_p: float = 0.3
    tokenizer: Optional[Any] = None  # 分词器
    history: Optional[List[Tuple[str, str]]] = None  # 历史对话
    model: Optional[Any] = None
    debug_mode: bool = False  # 调试模式标志
    device: Optional[str] = None  # 设备类型（cuda或cpu）

    def __init__(self, debug_mode=False):
        super().__init__()
        self.debug_mode = debug_mode

    @property # 装饰器，将方法转换为属性
    def _llm_type(self):
        return "deepseek_local_llm"
    
    def load_model(self, modelPath: str):
        print(f"开始加载模型，从路径: {modelPath}")
        print(f"PyTorch是否可用CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"当前CUDA设备: {torch.cuda.get_device_name(0)}")
            print(f"可用GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
        
        # 确保模型路径存在并且是本地路径
        model_path = Path(modelPath)
        if not model_path.exists():
            raise ValueError(f"模型路径不存在: {modelPath}")
        
        print(f"使用本地模型路径: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=True,
            local_files_only=True  # 确保只使用本地文件
        )
        print("分词器加载完成")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            use_cache=True,
            attn_implementation="eager",  # 使用eager实现而不是sdpa
            device_map=device,  # 显式指定设备
            local_files_only=True  # 确保只使用本地文件
        )
        print("模型加载完成")
        print(f"模型类型: {type(model).__name__}")
        
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        print("模型已准备好推理")
    
    def _call(self, prompt, config={}, history=[]):
        return self.invoke(prompt, config, history)
    
    def invoke(self, prompt, config={}, history=[]):

        if not self.tokenizer:
            raise ValueError("tokenizer is not loaded")
        if not self.model:
            raise ValueError("model is not loaded")
        if not isinstance(prompt, str):
            prompt = prompt.to_string()

        print("正在处理输入...")
        # 处理历史对话
        if history:
            full_prompt = ""
            for h_prompt, h_response in history:
                full_prompt += f"用户: {h_prompt}\n助手: {h_response}\n"
            full_prompt += f"用户: {prompt}\n助手: "
        else:
            full_prompt = f"用户: {prompt}\n助手: "
        
        # 编码输入
        print("正在编码输入...")
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        # 使用generate方法
        print("开始生成回复，这可能需要一些时间...")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,  # 缩短生成长度，加快测试
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        
        print("生成完成，正在解码...")
        # 解码输出，只取新生成的部分
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 更新历史
        if history is None:
            history = []
        history.append((prompt, response))
        self.history = history
        
        print("处理完毕!")
        return AIMessage(content=response)
    
    def stream(self, prompt, config={}, history=[]):

        if not self.tokenizer:
            raise ValueError("tokenizer is not loaded")
        if not self.model:
            raise ValueError("model is not loaded")
        if not isinstance(prompt, str):
            prompt = prompt.to_string()

        system_instruction = "你是一个AI助手，请直接回答用户的问题，不要假设自己是用户或其他角色，不要创建对话。"
        prompt = f"系统提示：{system_instruction}\n\n{prompt}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text
         
    
# 正常模式
llm = deepseek_local_llm()
llm.load_model(r"D:\ai\study\DeepSeek-R1-Distill-Qwen-1.5B")

output_parser = StrOutputParser()
prompt_template = ChatPromptTemplate.from_template("{text}")
chain = prompt_template | llm | output_parser

print("\n测试使用链式调用流式输出：")
for chunk in chain.stream({"text": "你好，给我讲个笑话"}):
    print(chunk, end="", flush=True)
print("\n链式流式输出测试完成")


    

        


