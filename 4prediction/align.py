import os
import pandas as pd
import time
import logging
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
# 添加 PyMOL 路径
sys.path.append(
    'C:/Users/15098/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0/LocalCache/local-packages/Python38/site-packages')
from pymol import cmd
from psico.fitting import tmalign

# 定义目录和文件名模式
input_dir = '../0storage/2selected_gomc'  # 输入文件目录
output_csv_dir = '../0storage/3selected_gomc_type_csv'  # 按类型划分的CSV文件保存目录
output_pdb_dir = '../0storage/4selected_gomc_type_pdb'  # 预测的PDB文件保存目录
reference_dir = '../0storage/5reference_sbp'  # 参考PDB文件目录
output_tmscore_dir = '../0storage/6selected_gomc_tm-score'  # TM-score结果保存目录
file_pattern = 'selected_gomc_{}_predicted_results.csv'

# 自定义起始和终止的n
start_n = 1100  # 起始编号
end_n = 1100  # 终止编号

# 确保输出目录存在
os.makedirs(output_csv_dir, exist_ok=True)
os.makedirs(output_pdb_dir, exist_ok=True)
os.makedirs(output_tmscore_dir, exist_ok=True)

# 配置日志
log_file = os.path.join(output_pdb_dir, 'prediction_errors.log')
logging.basicConfig(filename=log_file, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# 清理文件名中的非法字符
def clean_filename(filename):
    illegal_chars = '<>:"/\\|?*'
    for char in illegal_chars:
        filename = filename.replace(char, '_')
    return filename

# 使用ESM API预测蛋白质结构，支持重试机制
def predict_structure_with_api(sequence, max_retries=3, retry_delay=10):
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    for attempt in range(max_retries):
        try:
            response = requests.post(url, data=sequence, timeout=60)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed for sequence: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"API请求失败: {response.status_code} - {response.text}")

# 预测并保存单个序列的结构
def predict_and_save(gene_id, sequence, predicted_type):
    try:
        pdb_data = predict_structure_with_api(sequence)
        type_pdb_dir = os.path.join(output_pdb_dir, predicted_type)
        os.makedirs(type_pdb_dir, exist_ok=True)
        output_path = os.path.join(type_pdb_dir, f"{gene_id}.pdb")
        with open(output_path, "w") as f:
            f.write(pdb_data)
    except Exception as e:
        error_message = f"Error processing {gene_id}: {str(e)}"
        logging.error(error_message)  # 记录错误到日志文件
        raise Exception(error_message)

# 批量使用API预测蛋白质结构
def batch_predict_with_api(df, max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index, row in df.iterrows():
            gene_id = row['Gene ID']
            sequence = row['Protein Sequence']
            predicted_type = row['Predicted_Type']
            futures.append(executor.submit(predict_and_save, gene_id, sequence, predicted_type))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Predicting structures"):
            try:
                future.result()
            except Exception as e:
                print(str(e))  # 打印错误信息到控制台

# 加载 PDB 文件并计算 RMSD 和 TM-score
def calculate_scores(ref_pdb, candi_pdb):
    try:
        # 加载参考结构和待比对结构
        cmd.load(ref_pdb, "ref")
        cmd.load(candi_pdb, "candi")

        # 使用 super 命令进行结构比对，计算 RMSD
        rmsd = cmd.super("candi", "ref")  # RMSD 是一个浮点数

        # 使用 psico 的 tmalign 命令计算 TM-score
        tmscore = tmalign("candi", "ref", object="aln")  # object="aln" 用于存储对齐结果

        # 删除加载的结构以释放内存
        cmd.delete("ref")
        cmd.delete("candi")

        return rmsd, tmscore  # 返回 RMSD 和 TM-score
    except Exception as e:
        print(f"Error comparing {ref_pdb} and {candi_pdb}: {e}")
        return None, None

# 对每个预测的PDB文件与参考PDB文件进行比对
def compare_with_reference(predicted_type, original_df):
    # 清理 predicted_type 中的非法字符
    predicted_type = clean_filename(predicted_type)

    # 生成路径
    type_pdb_dir = os.path.join(output_pdb_dir, predicted_type)
    type_ref_dir = os.path.join(reference_dir, predicted_type)
    output_tmscore_path = os.path.join(output_tmscore_dir, f"{predicted_type}_with_scores.csv")

    # 检查 PDB 目录是否存在
    if not os.path.exists(type_pdb_dir):
        print(f"警告: PDB 目录 {type_pdb_dir} 不存在，跳过 {predicted_type}。")
        return

    # 检查参考目录是否存在
    if not os.path.exists(type_ref_dir):
        print(f"警告: 参考目录 {type_ref_dir} 不存在，跳过 {predicted_type}。")
        return

    # 获取所有PDB文件
    try:
        candi_pdb_files = [os.path.join(type_pdb_dir, f) for f in os.listdir(type_pdb_dir) if f.endswith(".pdb")]
    except FileNotFoundError:
        print(f"错误: PDB 目录 {type_pdb_dir} 不存在，跳过 {predicted_type}。")
        return

    ref_pdb_files = [os.path.join(type_ref_dir, f) for f in os.listdir(type_ref_dir) if f.endswith(".pdb")]

    results = []
    for candi_pdb in candi_pdb_files:
        # 从文件名中提取 Gene ID（假设文件名格式为 {Gene ID}.pdb）
        gene_id = os.path.splitext(os.path.basename(candi_pdb))[0]

        # 在原始数据中查找对应的行
        original_row = original_df[original_df['Gene ID'] == gene_id]
        if original_row.empty:  # 如果没有找到对应的 Gene ID
            print(f"警告: 未找到 Gene ID {gene_id} 的原始数据，跳过。")
            continue

        max_tmscore = -1
        best_rmsd = None
        best_ref_pdb = None

        for ref_pdb in ref_pdb_files:
            rmsd, tmscore = calculate_scores(ref_pdb, candi_pdb)
            if rmsd is not None and tmscore is not None and tmscore > max_tmscore:
                max_tmscore = tmscore
                best_rmsd = rmsd
                best_ref_pdb = os.path.basename(ref_pdb)

        if max_tmscore > 0.5:
            # 将原始数据和计算结果合并
            results.append({
                **original_row.iloc[0].to_dict(),  # 原始数据
                "Reference PDB": best_ref_pdb,
                "RMSD": best_rmsd,
                "TM-score": max_tmscore
            })

    # 保存结果到CSV
    if results:
        if os.path.exists(output_tmscore_path):
            existing_df = pd.read_csv(output_tmscore_path)
            combined_df = pd.concat([existing_df, pd.DataFrame(results)], ignore_index=True)
            combined_df.to_csv(output_tmscore_path, index=False)
        else:
            df = pd.DataFrame(results)
            df.to_csv(output_tmscore_path, index=False)
        print(f"Saved complete results for {predicted_type} to {output_tmscore_path}")

# 主逻辑
for n in range(start_n, end_n + 1):
    print(f"当前处理 n = {n}")  # 打印当前处理的 n
    file_name = file_pattern.format(n)
    file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(file_path):
        print(f"文件 {file_name} 不存在，跳过。")
        continue

    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 清理 Predicted_Type 列
    df['Predicted_Type'] = df['Predicted_Type'].apply(clean_filename)

    # ①按 Predicted_Type 划分并保存到 3selected_gomc_type_csv
    for predicted_type, group in df.groupby('Predicted_Type'):
        # 清理 predicted_type 中的非法字符
        predicted_type = clean_filename(predicted_type)

        # 生成路径
        type_csv_path = os.path.join(output_csv_dir, f"{predicted_type}.csv")

        # 确保目录存在
        os.makedirs(os.path.dirname(type_csv_path), exist_ok=True)

        # 保存数据
        if os.path.exists(type_csv_path):
            existing_df = pd.read_csv(type_csv_path)
            combined_df = pd.concat([existing_df, group], ignore_index=True)
            combined_df.to_csv(type_csv_path, index=False)
        else:
            group.to_csv(type_csv_path, index=False)
        print(f"Saved {predicted_type} data to {type_csv_path}")

    # ②对每一行数据进行结构预测并保存到 4selected_gomc_type_pdb
    batch_predict_with_api(df)

    # ③对每个预测的PDB文件与参考PDB文件进行比对，保存TM-score > 0.5的结果
    for predicted_type in df['Predicted_Type'].unique():
        compare_with_reference(predicted_type, df)  # 传入原始数据

print("所有操作完成。")