from transformers import BertTokenizer, BertModel
import torch
import re

class ProteinEmbeddingGenerator:
    def __init__(self, model_dir='protBERT', device=None):
        """
        :param model_dir: Directory path for ProtBERT model and tokenizer.
        :param device: Computing device, defaults to 'cuda' if available, otherwise 'cpu'.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_dir).to(self.device)

    def preprocess_sequence(self, sequence):
        """
        Preprocess protein sequence to ensure each amino acid is separated by space.
        :param sequence: Input protein sequence (continuous string or space-separated string).
        :return: Preprocessed protein sequence.
        """
        if ' ' not in sequence:
            sequence = ' '.join(list(sequence))
        sequence = re.sub(r"[UZOB]", "X", sequence)  # 替换无法识别的氨基酸
        return sequence

    def embedding(self, sequences):
        """
        为输入的蛋白质序列生成嵌入。
        :param sequences: 包含蛋白质序列的列表。
        :return:
            - global_embeddings: 形状为 (batch_size, dim) 的全局嵌入（[CLS]）。
            - local_embeddings: 形状为 (batch_size, length, dim) 的局部嵌入（[AA]，不包含 [SEP]）。
            - mask: 形状为 (batch_size, length) 的掩码，1 表示有效 token，0 表示填充 token。
        """
        # 预处理所有序列
        processed_sequences = [self.preprocess_sequence(seq) for seq in sequences]

        # 编码蛋白质序列
        encoded_input = self.tokenizer(
            processed_sequences,
            return_tensors='pt',
            padding=True,  # 自动填充到最大长度
            truncation=False  # 不截断
        ).to(self.device)

        # 使用模型生成嵌入
        with torch.no_grad():
            outputs = self.model(**encoded_input)

        # 获取最后一个隐藏层的状态
        full_embeddings = outputs.last_hidden_state.detach().cpu()  # 形状为 (batch_size, max_length, dim)

        # 提取全局嵌入（[CLS] token）
        global_embeddings = full_embeddings[:, 0, :]  # 形状为 (batch_size, dim)

        # 提取局部嵌入（[AA] tokens），去掉 [CLS] 和 [SEP]
        sep_token_id = self.tokenizer.sep_token_id  # 获取 [SEP] 的 token ID
        input_ids = encoded_input['input_ids'].cpu()  # 获取 input_ids

        # 找到 [SEP] 的位置
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[1]

        # 提取局部嵌入，去掉 [SEP]
        local_embeddings = []
        local_mask = []
        for i in range(full_embeddings.shape[0]):
            # 从第 1 个位置到 [SEP] 的前一个位置
            local_embeddings.append(full_embeddings[i, 1:sep_positions[i], :])
            local_mask.append(input_ids[i, 1:sep_positions[i]] != self.tokenizer.pad_token_id)
        local_embeddings = torch.nn.utils.rnn.pad_sequence(local_embeddings, batch_first=True)  # 填充到相同长度
        local_mask = torch.nn.utils.rnn.pad_sequence(local_mask, batch_first=True)  # 填充到相同长度

        return global_embeddings, local_embeddings, local_mask