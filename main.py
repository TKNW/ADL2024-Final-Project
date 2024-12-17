from base import Agent
from execution_pipeline import main
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch
import faiss
import re
import random
from colorama import Fore, Style
from utils import strip_all_lines

RAG_NUMBER = 16

class ClassificationAgent(Agent):
    """
    An agent that classifies text into one of the labels in the given label set.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize your LLM here
        """
        # TODO
        print(config)
        super().__init__(config)
        self.llm_config = config
        if config['use_8bit']:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_has_fp16_weight=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                quantization_config=quantization_config,
                device_map=config["device"]
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.float16,
                device_map=config["device"]
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.model.eval()
        # 前一次的問題和答案
        self.last_input = ""
        self.sentencetrans_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2',device=config["device"])
        test = self.sentencetrans_model.encode("This is a sentece.")
        self.index = faiss.IndexFlatL2(test.shape[0])
        self.id_doc = {}

    def get_prompt(self, option_text: str, text: str) -> str:
        prompt = f"""\
        Act as a medical doctor and diagnose the patient based on the following patient profile:

        {text}

        All possible diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <number>. <diagnosis>):
        {option_text}
        
        Now, take a deep breath, think step by step, and provide your answer in the following valid JSON format:
        {{
            "reason": "<think step by step>",
            "answer": "<number>. <diagnosis>",
        }}
        DO NOT include other information. Make sure your output is in 100 words.""".strip()
        return strip_all_lines(prompt)

    def get_system_prompt(self, texts) -> str:
        RAGtext = ""
        if self.index.ntotal >= RAG_NUMBER:
            RAGtext = "Here are some past cases for your reference. Each case includes profile, the reason and the answer.\n"
            textlist = self.search_and_restore(texts, RAG_NUMBER)
            i = 1
            for text, num in textlist:
                RAGtext = RAGtext + f"\nCase{i}.\n{text[0]}"
                i += 1
        system_prompt = f"""\
        Act as a professional medical doctor that can diagnose the patient based on the patient profile. {RAGtext}
        Provide your output in the following valid JSON format:
        {{
            "reason": "<think step by step>",
            "answer": "<number>. <diagnosis>",
        }}
        DO NOT include other information. Make sure your output is in 100 words.""".strip()
        return strip_all_lines(system_prompt)
    
    def extract_label(self, pred_text: str, label2desc: dict[str, str]) -> str:
        numbers = re.findall(pattern=r"(\d+)\.", string=pred_text)
        if len(numbers) == 1:
            number = numbers[0]
            if int(number) in label2desc:
                prediction = number
            else:
                print(Fore.RED + f"Prediction {pred_text} not found in the label set. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        else:
            if len(numbers) > 1:
                print(Fore.YELLOW + f"Extracted numbers {numbers} is not exactly one. Select the first one." + Style.RESET_ALL)
                prediction = numbers[0]
            else:
                print(Fore.RED + f"Prediction {pred_text} has no extracted numbers. Randomly select one." + Style.RESET_ALL)
                prediction = random.choice(list(label2desc.keys()))
        return str(prediction)
        
    # 由GPT協助生成
    # 查詢相似的句子
    def search_and_restore(self, query, top_k=5):
        # 查詢的編碼向量
        query_embedding = self.sentencetrans_model.encode([query])
        # 在 FAISS 中查詢
        distances, indices = self.index.search(query_embedding, top_k)
        # 通過索引字典還原字串
        results = [(self.id_doc[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        return results
    def __call__(
        self,
        label2desc: dict[str, str],
        text: str
    ) -> str:
        """
        Classify the text into one of the labels.

        Args:
            label2desc (dict[str, str]): A dictionary mapping each label to its description.
            text (str): The text to classify.

        Returns:
            str: The label (should be a key in label2desc) that the text is classified into.

        For example:
        label2desc = {
            "apple": "A fruit that is typically red, green, or yellow.",
            "banana": "A long curved fruit that grows in clusters and has soft pulpy flesh and yellow skin when ripe.",
            "cherry": "A small, round stone fruit that is typically bright or dark red.",
        }
        text = "The fruit is red and about the size of a tennis ball."
        label = "apple" (should be a key in label2desc, i.e., ["apple", "banana", "cherry"])
        """
        # TODO
        option_text = '\n'.join([f"{str(k)}. {v}" for k, v in label2desc.items()])
        input_text = self.get_prompt(option_text, text)
        system_prompt = self.get_system_prompt(input_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        text_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_chat], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.llm_config["max_tokens"],
            do_sample=self.llm_config["do_sample"],
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.update_log_info(log_data={
            "num_input_tokens": len(system_prompt + input_text),
            "num_output_tokens": len(self.tokenizer.encode(output)),
            "num_shots": str(0 if self.index.ntotal < RAG_NUMBER else RAG_NUMBER),
            "input_pred": input_text,
            "output_pred": output,
        })

        match = re.search(r'"answer":\s*"([^"]+)"', output)
        if match:
            output = match.group(1)
        
        pred = self.extract_label(output, label2desc)
        self.last_input = text + '\nResponse: \n' + output
        return pred
        # raise NotImplementedError

    def update(self, correctness: bool) -> bool:
        """
        Update your LLM agent based on the correctness of its own prediction at the current time step.

        Args:
            correctness (bool): Whether the prediction is correct.

        Returns:
            bool: Whether the prediction is correct.
        """
        # TODO
        # 正確的話，embedding後直接存
        if correctness:
            # print(self.last_input)
            RAGinput = [self.last_input]
            RAGinput_embeddings = self.sentencetrans_model.encode(RAGinput)
            self.index.add(RAGinput_embeddings)
            start_index = len(self.id_doc)
            self.id_doc[start_index] = RAGinput
        return correctness
        # raise NotImplementedError

class SQLGenerationAgent(Agent):
    """
    An agent that generates SQL code based on the given table schema and the user query.
    """
    
    def get_system_prompt(self) -> str:
        system_prompt = f"""\
        ### Role
        # Act as a professional programmer.
        ### Task
        # You will be given a table schema and a user query. 
        # Complete sqlite SQL query only and with no explanation.
        ### Output Format
        # Return your answer in the following format:
        ```sql\n<your_SQL_code>\n```
        """
        return strip_all_lines(system_prompt)
    
    def get_prompt(self, table_schema: str, user_query: str) -> str:
        RAGtext = ""
        if self.index.ntotal >= 1:
            RAGtext = "# The following are some related SQL code for your reference, include question and answer:\n"
            textlist = self.search_and_restore(self.last_skill, RAG_NUMBER)
            i = 1
            for text, num in textlist:
                RAGtext = RAGtext + f"\n{text[0]}\n"
                i += 1
        prompt = f"""\
        # You are performing the text-to-SQL task.

        {RAGtext}

        # Now, it's your turn.

        SQL schema: {table_schema}
        Question: {user_query}

        # Complete sqlite SQL query only and with no explanation, in the following format:
        ```sql\n<your_SQL_code>\n```
        """
        return strip_all_lines(prompt)

    # 生成skill-based描述的system prompt
    def get_skill_prompt(self) -> str:
        system_prompt = f"""\
        # Generate the needed skills to solve the task on the database schema.

        ### Task
        # You will be given a table schema and a user query. 
        # Think carefully and thoroughly before providing the answer.

        ## Output Format
        # Return your answer in the following format:
        To solve this task in the database, we need to <your answer>
        """
        return strip_all_lines(system_prompt)

    
    # 請模型生成前問題描述對應的skill-based描述
    def generate_skill_sentence(self, schema:str, question:str)->str:
        prompt = F"""\
        # Generate the needed skills to solve the task on the database schema.

        # Here are some example:

        ## Database schema:
        allergy_type (allergy, allergytype)
        has_allergy (stuid, allergy)
        student (stuid, lname, fname, age, sex, major, advisor,
        city_code)
        ## Task: Show all majors.
        To solve this task in the database, we need to select distinct values in the column.

        ## Database schema:
        department (department_id, name, creation, ranking,
        budget_in_billions, num_employees)
        head (head_id, name, born_state, age)
        management (department_id, head_id, temporary_acting)
        ## Task: What are the maximum and minimum budget of the departments?
        To solve this task in the database, we need to return the minimum value and the maximum value in one column.

        # Here are the table and the task:
        ## Table:
        {schema}
        ##Task:\n{question}
        # Return your answer in the following format, do not output any other information:
        To solve this task in the database, we need to <your answer>"""
        messages = [
                {"role": "system", "content": self.get_skill_prompt()},
                {"role": "user", "content": strip_all_lines(prompt)}
            ]
        text_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text_chat], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.llm_config["max_tokens"],
            do_sample=self.llm_config["do_sample"],
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output

    # 由GPT協助生成
    # 查詢相似的句子
    def search_and_restore(self, query, top_k=5):
        if self.index.ntotal == 0:
            return []
        # 小於設定的值，就回傳目前全部的
        if self.index.ntotal < top_k:
            top_k = self.index.ntotal
        # 查詢的編碼向量
        query_embedding = self.sentencetrans_model.encode([query])
        # L2正規化，把向量轉成0~1，這樣search才會等價於餘弦相似度
        faiss.normalize_L2(query_embedding)
        # 在 FAISS 中查詢
        distances, indices = self.index.search(query_embedding, top_k)
        # 通過索引字典還原字串
        results = [(self.id_doc[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
        # 最接近的範例放最後面 所以Reverse一下
        results.reverse()
        return results
    
    # 由GPT協助生成
    # 將schema文法整理成下面格式：
    # department (department_id, name, creation, ranking,
    # budget_in_billions, num_employees)
    # head (head_id, name, born_state, age)
    # management (department_id, head_id, temporary_acting)
    # 目前不使用
    def get_schema(self, schema:str) ->str:
        pattern = r"(?=CREATE TABLE)"
        tables = re.split(pattern, schema.strip(), flags=re.IGNORECASE)
        result = []
        for table in tables:
            if not table.strip(): 
                continue
            match = re.match(r"CREATE TABLE\s+(`[\w_]+`|\"[\w_]+\"|[\w_]+)\s*\((.*?)\)", table.strip(), re.DOTALL)
            if match:
                table_name = match.group(1)
                columns = match.group(2)
                column_names = [line.split()[0] for line in columns.splitlines() if line.strip()]
                formatted_table = f"{table_name} ({', '.join(column_names)})"
                result.append(formatted_table)
        output = "\n".join(result)
        return output
    
    def __init__(self, config: dict) -> None:
        """
        Initialize your LLM here
        """
        # TODO
        print(config)
        super().__init__(config)
        self.llm_config = config
        if config['use_8bit']:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_has_fp16_weight=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                quantization_config=quantization_config,
                device_map=config["device"]
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype=torch.float16,
                device_map=config["device"]
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.model.eval()
        # 最後一次的output
        self.last_input = ""
        # 最後一次的skill based描述
        self.last_skill = ""
        # embedding用的模型
        self.sentencetrans_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2',device=config["device"])
        test = self.sentencetrans_model.encode("This is a sentece.")
        self.index = faiss.IndexFlatIP(test.shape[0])
        self.id_doc = {}
        #raise NotImplementedError
    
    def __call__(
        self,
        table_schema: str,
        user_query: str
    ) -> str:
        """
        Generate SQL code based on the given table schema and the user query.

        Args:
            table_schema (str): The table schema.
            user_query (str): The user query.

        Returns:
            str: The SQL code that the LLM generates.
        """
        # TODO: Note that your output should be a valid SQL code only.
        # table_schema = self.get_schema(table_schema)
        # 這個要先做，不然查的時候跟存的時候會沒東西
        self.last_skill = self.generate_skill_sentence(table_schema, user_query)

        input_text = self.get_prompt(table_schema, user_query)
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": input_text}
        ]
        text_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text_chat], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.llm_config["max_tokens"],
            do_sample=self.llm_config["do_sample"],
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        self.update_log_info(log_data={
            "num_input_tokens": len(self.tokenizer.encode(self.get_system_prompt() + input_text)),
            "num_output_tokens": len(self.tokenizer.encode(output)),
            "num_shots": str(0 if self.index.ntotal < RAG_NUMBER else RAG_NUMBER),
            "input_pred": user_query,
            "output_pred": output,
        })
        self.last_input = "Question:\n" + user_query + "\nAnswer:\n" + output 
        output = self.parse_sql(output)
        return output

    def update(self, correctness: bool) -> bool:
        """
        Update your LLM agent based on the correctness of its own SQL code at the current time step.
        """
        # TODO
        # 如果正確的話，把skill based的描述當成RAG的Key，放進faiss的index
        if correctness:
            RAGinput = [self.last_skill]
            RAGinput_embeddings = self.sentencetrans_model.encode(RAGinput)
            faiss.normalize_L2(RAGinput_embeddings)
            self.index.add(RAGinput_embeddings)
            start_index = len(self.id_doc)
            self.id_doc[start_index] = [self.last_input]
        return correctness

    # 從self_steamicl.py借來的，取出sql code
    @staticmethod
    def parse_sql(pred_text: str) -> str:
        """
        Parse the SQL code from the LLM's response.
        """
        pattern = r"```sql([\s\S]*?)```"
        match = re.search(pattern, pred_text)
        if match:
            sql_code = match.group(1)
            sql_code = sql_code.strip()
            return sql_code
        else:
            print(Fore.RED + "No SQL code found in the response" + Style.RESET_ALL)
            sql_code = pred_text
        return sql_code
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    from execution_pipeline import main

    parser = ArgumentParser()
    parser.add_argument('--bench_name', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.bench_name.startswith("classification"):
        agent_name = ClassificationAgent
        max_token = 160
    elif args.bench_name.startswith("sql_generation"):
        agent_name = SQLGenerationAgent
        max_token = 512
    else:
        raise ValueError(f"Invalid benchmark name: {args.bench_name}")

    bench_cfg = {
        'bench_name': args.bench_name,
        'output_path': args.output_path,
        'debug':args.debug
    }
    config = {
        # TODO: specify your configs for the agent here
        'model_name': "Qwen/Qwen2.5-7B-Instruct",
        'exp_name': f'{args.bench_name}_json_inference_redo2_RAG16_{"Qwen/Qwen2.5-7B-Instruct"}_8bit-{True}',
        'bench_name': args.bench_name,
        'use_8bit': True,
        'device': "cuda:0",
        'do_sample': False,
        'max_tokens': max_token,
    }
    agent = agent_name(config)
    main(agent, bench_cfg, use_wandb=args.use_wandb, wandb_name = config['exp_name'], debug=args.debug, debug_samples=100)

