import os
import re
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
sys.path.append('../2model_train')
from model import GlobalLocalNetwork
from embedding import ProteinEmbeddingGenerator
from tqdm import tqdm
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import warnings

warnings.filterwarnings("ignore")


class ProteinDataset:

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class Predictor:
    def __init__(self, model_dir='model_dir', global_dim=1024, local_dim=1024, num_classes=54, model_path='model_path'):
        """
        :param model_dir: Directory path for ProtBERT model and tokenizer.
        :param global_dim: Dimension of global features.
        :param local_dim: Dimension of local features.
        :param num_classes: Number of classes for classification task.
        :param model_path: Path to trained model weights.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('现在使用的设备:', self.device)
        self.embedding_generator = ProteinEmbeddingGenerator(model_dir, self.device)
        self.model = GlobalLocalNetwork(global_dim, local_dim, num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))  # 加载模型权重
        self.model.eval()
        self.num_classes = num_classes

        # Define mapping between Predicted_Label and Predicted_PSS_ID/Predicted_Type
        self.label_to_pss_id = {
            1: "PS001", 2: "PS002", 3: "PS003", 4: "PS003431", 5: "PS003436",
            6: "PS003438", 7: "PS004", 8: "PS005", 9: "PS006", 10: "PS007",
            11: "PS008", 12: "PS009", 13: "PS011", 14: "PS014", 15: "PS015",
            16: "PS019", 17: "PS020", 18: "PS021", 19: "PS022", 20: "PS024",
            21: "PS026", 22: "PS027", 23: "PS029", 24: "PS030", 25: "PS031",
            26: "PS032", 27: "PS033", 28: "PS034", 29: "PS035", 30: "PS036",
            31: "PS037", 32: "PS038", 33: "PS039", 34: "PS040", 35: "PS041",
            36: "PS042", 37: "PS047", 38: "PS048", 39: "PS049", 40: "PS050",
            41: "PS051", 42: "PS052", 43: "PS054", 44: "PS055", 45: "PS056",
            46: "PS057", 47: "PS058", 48: "PS059", 49: "PS060", 50: "PS063",
            51: "PS064", 52: "PS065", 53: "PS067"
        }

        self.label_to_type = {
            1: "ABD-derived affinity protein", 2: "Abdurin", 3: "Abl/Hck SH3 domain",
            4: "Interleukin mimic", 5: "Miniprotein LCB1", 6: "Miniprotein LCB3", 7: "Affibody", 8: "Affilin",
            9: "Affimer", 10: "Affitin", 11: "Alphabody", 12: "Alpha-Rep protein",
            13: "Anticalin", 14: "Avimer", 15: "Beta Roll domain", 16: "CBM-based binder",
            17: "Centyrin", 18: "Chaperonin 10-based binder", 19: "CI2-based binder",
            20: "Cytochrome b562-based binder", 21: "DArmRP", 22: "DARPin",
            23: "Defensin A-based binder", 24: "Designed TPR protein", 25: "Diabody",
            26: "EVH1 domain", 27: "Evibody", 28: "Fab", 29: "Fynomer", 30: "GCN4-based binder",
            31: "Glubody", 32: "Gp2-based binder", 33: "Human VH dAb", 34: "I-body",
            35: "Im9-based binder", 36: "Knottin", 37: "Monobody", 38: "Nanobody",
            39: "Neocarzinostatin-based binder", 40: "OBody", 41: "Peptide aptamer",
            42: "PHD finger domain", 43: "PVIII-based binder", 44: "Repebody",
            45: "RPtag", 46: "scFv", 47: "SH2 domain", 48: "SWEEPin", 49: "Telobody",
            50: "Transferrin-based binder", 51: "VL dAb", 52: "vNAR", 53: "WW domain"
        }

    def predict(self, csv_path, output_csv_path='predictions.csv', batch_size=32):

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Protein Sequence'])  
        df['Protein Sequence'] = df['Protein Sequence'].astype(str)
        sequences = df['Protein Sequence'].tolist()

        dataset = ProteinDataset(sequences)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_preds = []
        all_probs = []
        with torch.no_grad():
            for batch_sequences in tqdm(dataloader, desc="predicting"):
                global_embeddings, local_embeddings, mask = self.embedding_generator.embedding(batch_sequences)

                global_embeddings = global_embeddings.to(self.device)
                local_embeddings = local_embeddings.to(self.device)
                mask = mask.to(self.device)

                outputs = self.model(global_embeddings, local_embeddings, mask)

                # 将 logits to probabilities using softmax function
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs)

        # results save
        df['Predicted_Label'] = all_preds
        df['Max_Probability'] = [max(prob) for prob in all_probs]

        df['Predicted_PSS_ID'] = df['Predicted_Label'].map(self.label_to_pss_id)
        df['Predicted_Type'] = df['Predicted_Label'].map(self.label_to_type)

        for i in range(self.num_classes):
            df[f'Class_{i}_Probability'] = [prob[i] for prob in all_probs]

        print("remove sequence with X...")
        df = df[~df['Protein Sequence'].str.contains('X', case=False, na=False)]

        #for label=0, removal
        print("remove line with Predicted_Label=0...")
        df = df[df['Predicted_Label'] != 0]

        columns_to_keep = ['SRR ID', 'Gene ID', 'Protein Sequence', 'Predicted_Label', 'Max_Probability', 'Predicted_PSS_ID',
                          'Predicted_Type']
        df = df[columns_to_keep]

        def calculate_sequence_length(sequence):
            return len(sequence)

        def calculate_molecular_weight(sequence):
            analysis = ProteinAnalysis(sequence)
            return analysis.molecular_weight() / 1000

        print("calculating Sequence Length和Molecular Weight...")
        tqdm.pandas()  
        df['Sequence Length'] = df['Protein Sequence'].progress_apply(calculate_sequence_length)
        df['Molecular Weight'] = df['Protein Sequence'].progress_apply(calculate_molecular_weight)

        #basic filter via the information from PP.xlsx
        pp_df = pd.read_excel('../0storage/PP.xlsx')

        label_length_weight = pp_df[['Label', 'Sequence Length', 'Molecular Weight']]

        def is_within_range(row, label_length_weight):
            label = row['Predicted_Label']
            length = row['Sequence Length']
            weight = row['Molecular Weight']

            range_info = label_length_weight[label_length_weight['Label'] == label]
            if range_info.empty:
                return False

            length_range = range_info['Sequence Length'].values[0]
            weight_range = range_info['Molecular Weight'].values[0]

            if length_range == "Long" and weight_range == "Long":
                return True

            if length_range == "Long":
                try:
                    if isinstance(weight_range, str) and '-' in weight_range:
                        weight_min = float(weight_range.split('-')[0])
                        weight_max = float(weight_range.split('-')[1].replace('kDa', '').replace('kD', '').strip())
                    else:
                        weight_min = float(weight_range.replace('kDa', '').replace('kD', '').strip())
                        weight_max = weight_min
                except ValueError:
                    weight_min = 0
                    weight_max = float('inf')

                return weight_min <= weight <= weight_max

            if weight_range == "Long":
                try:
                    if isinstance(length_range, str) and ('+' in length_range or '*' in length_range):
                        length_min = 0 
                        length_max = float('inf')
                    else:
                        length_min = int(length_range)
                        length_max = int(length_range)
                except ValueError:
                    length_min = 0
                    length_max = float('inf')

                return length_min <= length <= length_max

            try:
                if isinstance(length_range, str) and ('+' in length_range or '*' in length_range):
                    length_min = 0
                    length_max = float('inf')
                else:
                    length_min = int(length_range)
                    length_max = int(length_range)
            except ValueError:
                length_min = 0
                length_max = float('inf')

            try:
                if isinstance(weight_range, str) and '-' in weight_range:
                    weight_min = float(weight_range.split('-')[0])
                    weight_max = float(weight_range.split('-')[1].replace('kDa', '').replace('kD', '').strip())
                else:
                    weight_min = float(weight_range.replace('kDa', '').replace('kD', '').strip())
                    weight_max = weight_min
            except ValueError:
                weight_min = 0
                weight_max = float('inf')

            return (length_min <= length <= length_max) and (weight_min <= weight <= weight_max)

        print("Comparing and filtering data...")
        df = df[df.progress_apply(is_within_range, axis=1, label_length_weight=label_length_weight)]

        # 保存结果
        print(f"save results to {output_csv_path}...")
        df.to_csv(output_csv_path, index=False)

        print("complete")


def predict_all_files_in_directory(input_dir, output_dir, start_idx, end_idx, batch_size=32):
    """
    遍历目录下的所有小 CSV 文件，并根据规定的起始和结束范围逐个进行预测处理。

    :param input_dir: 包含待预测 CSV 文件的目录路径。
    :param output_dir: 保存预测结果的目录路径。
    :param start_idx: 文件名的起始编号（包含）。
    :param end_idx: 文件名的结束编号（包含）。
    :param batch_size: 批次大小。
    """
    os.makedirs(output_dir, exist_ok=True)

    all_files = os.listdir(input_dir)
    csv_files = [f for f in all_files if f.endswith('.csv')]

    csv_files.sort(key=lambda x: int(re.search(r'selected_gomc_(\d+)\.csv', x).group(1)))

    model_path = '../0data_save/model_weight/epoch_49_model.pth'
    predictor = Predictor(model_dir='D:/MiningG/1protBERT', global_dim=1024, local_dim=1024, num_classes=54, model_path=model_path)

    # 遍历所有 CSV 文件
    for csv_file in csv_files:
        # 提取文件名中的数字
        match = re.search(r'selected_gomc_(\d+)\.csv', csv_file)
        if not match:
            print(f"跳过文件 {csv_file}，文件名不符合格式要求。")
            continue

        file_idx = int(match.group(1))  # 提取文件名中的数字部分
        if start_idx <= file_idx <= end_idx:  # 检查是否在规定的范围内
            print(f"正在处理文件: {csv_file} (编号: {file_idx})")

            # 构建输入和输出路径
            input_csv_path = os.path.join(input_dir, csv_file)
            output_csv_path = os.path.join(output_dir, f"{os.path.splitext(csv_file)[0]}_predicted_results.csv")

            # 调用 Predictor 进行预测
            predictor.predict(csv_path=input_csv_path, output_csv_path=output_csv_path, batch_size=batch_size)
        else:
            print(f"跳过文件 {csv_file}，编号 {file_idx} 不在规定范围内。")

    print("所有文件处理完成！")


# 示例调用
input_directory = '../0storage/1select_gomc'  # 待预测文件的目录
output_directory = '../0storage/2selected_gomc'  # 预测结果的保存目录
start_index = 1100  # 起始编号
end_index = 1100  # 结束编号

# 遍历目录并处理文件
predict_all_files_in_directory(input_directory, output_directory, start_index, end_index, batch_size=16)