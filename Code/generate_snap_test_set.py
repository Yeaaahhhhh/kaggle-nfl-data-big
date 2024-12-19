import pandas as pd
import glob
import orjson
from multiprocessing import Pool, cpu_count
from functools import partial
import os

def process_one_tracking_file(tracking_file, player_play_df):
    """
    处理单个tracking_week文件，提取游戏中snap_time_elapse与对应的nflId信息。

    Args:
        tracking_file (str): tracking_week文件的路径
        player_play_df (pd.DataFrame): player_play.csv的完整数据

    Return:
        dict: { "gameId_playId": { snap_time_elapse: nflId } }
    """
    usecols = ['gameId','playId','frameId','nflId','event']
    dtype_map = {
        'gameId': 'int32',
        'playId': 'int32',
        'frameId': 'int32',
        'nflId': 'float32',
        'event': 'object'
    }
    frame_gap = 0.1
    try:
        df = pd.read_csv(tracking_file, usecols=usecols, dtype=dtype_map)
    except Exception as e:
        print(f"Error reading {tracking_file}: {e}")
        return {}

    # 按 gameId, playId 分组
    final_dict = {}

    for (gameId, playId), group in df.groupby(['gameId','playId']):
        # 查看本回合是否有pass_forward或handoff事件
        target_events = group[group['event'].isin(['pass_forward', 'handoff'])]
        ball_snap = group[group['event'] == 'ball_snap']

        if ball_snap.empty or target_events.empty:
            # 没有pass_forward或handoff事件,或者没有ball_snap，无法计算snap_time_elapse
            continue

        # 假设一个play中只能有一个snap对应的事件计算
        snap_frame = ball_snap['frameId'].values[0]
        # 选取第一个pass_forward或handoff事件
        target_frame = target_events['frameId'].iloc[0]

        snap_time_elapse = frame_gap * (target_frame - snap_frame)
        # 保留一位小数
        snap_time_elapse = round(snap_time_elapse, 1)

        # 在player_play.csv中找到对应行
        ppdf = player_play_df[(player_play_df['gameId']==gameId) & (player_play_df['playId']==playId)]

        if ppdf.empty:
            # 未在player_play找到对应数据，跳过
            continue

        # 根据wasTargettedReceiver判断
        receiver_info = ppdf[ppdf['wasTargettedReceiver'] == 1]
        if receiver_info.empty:
            # 如果没有被标记为wasTargettedReceiver的接球手，则看hadRushAttempt
            rush_info = ppdf[ppdf['hadRushAttempt'] == 1]
            if rush_info.empty:
                # 没有标记为rush的球员，也没有目标接球手
                continue
            else:
                # 使用rush_info的nflId
                nflId = rush_info['nflId'].iloc[0]
        else:
            # 使用receiver_info的nflId
            nflId = receiver_info['nflId'].iloc[0]

        # 将结果写入final_dict
        key = f"{gameId}_{playId}"
        if key not in final_dict:
            final_dict[key] = {}
        final_dict[key][str(snap_time_elapse)] = int(nflId) if not pd.isna(nflId) else None

    return final_dict

def merge_dicts(dict_list):
    """
    合并多个字典为一个字典。

    Args:
        dict_list (list): 字典列表

    Return:
        dict: 合并后的字典
    """
    merged = {}
    for d in dict_list:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if k in merged:
                # 如果同一个key重复出现，根据需求可以决定是否覆盖或者合并
                # 此处简单覆盖
                merged[k].update(v)
            else:
                merged[k] = v
    return merged

def main():
    # 假设 player_play.csv 和 tracking_week_*.csv 文件在当前目录下
    player_play_path = 'player_play.csv'
    tracking_pattern = 'tracking_week_*.csv'

    try:
        player_play_df = pd.read_csv(player_play_path, dtype={
            'gameId':'int32','playId':'int32','nflId':'int32','wasTargettedReceiver':'int8','hadRushAttempt':'int8'})
    except Exception as e:
        print(f"Error reading player_play.csv: {e}")
        return

    tracking_files = glob.glob(tracking_pattern)
    if not tracking_files:
        print(f"No files found for pattern '{tracking_pattern}'")
        return
    
    # 使用多进程处理
    num_processes = min(cpu_count(), len(tracking_files))
    pool = Pool(processes=num_processes)

    process_func = partial(process_one_tracking_file, player_play_df=player_play_df)

    try:
        results = pool.map(process_func, tracking_files)
    except Exception as e:
        print(f"Error during multiprocessing: {e}")
        pool.close()
        pool.join()
        return

    pool.close()
    pool.join()

    final_dict = merge_dicts(results)

    # 使用orjson输出json文件
    # 要求格式工整，且有换行与缩进
    # orjson默认不支持直接格式化缩进输出，但可以用option=orjson.OPT_INDENT_2
    # 注意：使用option=orjson.OPT_INDENT_2会产生类似：
    # {
    #   "key": {
    #     "subkey": value
    #   }
    # }
    # 的格式

    json_bytes = orjson.dumps(final_dict, option=orjson.OPT_INDENT_2)

    output_path = os.path.join(os.getcwd(), 'snap_elapse_receiver.json')
    with open(output_path, 'wb') as f:
        f.write(json_bytes)

    print("snap_elapse_receiver.json has been generated.")

if __name__ == "__main__":
    main()
