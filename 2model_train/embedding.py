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
        sequence = re.sub(r"[UZOB]", "X", sequence)
        return sequence

    def embedding(self, sequences):
        processed_sequences = [self.preprocess_sequence(seq) for seq in sequences]

        encoded_input = self.tokenizer(
            processed_sequences,
            return_tensors='pt',
            padding=True,  
            truncation=False  
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded_input)

        full_embeddings = outputs.last_hidden_state.detach().cpu()  

        global_embeddings = full_embeddings[:, 0, :]  

        sep_token_id = self.tokenizer.sep_token_id  
        input_ids = encoded_input['input_ids'].cpu()  

        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[1]

        local_embeddings = []
        local_mask = []
        for i in range(full_embeddings.shape[0]):
            local_embeddings.append(full_embeddings[i, 1:sep_positions[i], :])
            local_mask.append(input_ids[i, 1:sep_positions[i]] != self.tokenizer.pad_token_id)
        local_embeddings = torch.nn.utils.rnn.pad_sequence(local_embeddings, batch_first=True)  
        local_mask = torch.nn.utils.rnn.pad_sequence(local_mask, batch_first=True)  

        return global_embeddings, local_embeddings, local_mask
