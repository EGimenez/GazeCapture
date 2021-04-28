# The following script is meant to convert json gaze info about, handsets, ... to csv format

import os
import argparse
import telefonica_reseach.config as config
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from telefonica_reseach.utils.multiprocessing.MultiProcessingFramework import MultiProcessManager, Processor

apple_device_data = pd.read_csv(config.apple_device_data_path)


def process_appleFace(user: Path):
    # H, W, X, Y, IsValid
    with open(user / 'appleFace.json') as json_file:
        data = json.load(json_file)

    df = pd.DataFrame()
    df['face_H'] = data['H']
    df['face_W'] = data['W']
    df['face_X'] = data['X']
    df['face_Y'] = data['Y']
    df['face_IsValid'] = data['IsValid']

    return df


def process_appleLeftEye(user: Path):
    # H, W, X, Y, IsValid
    with open(user / 'appleLeftEye.json') as json_file:
        data = json.load(json_file)

    df = pd.DataFrame()
    df['LeftEye_H'] = data['H']
    df['LeftEye_W'] = data['W']
    df['LeftEye_X'] = data['X']
    df['LeftEye_Y'] = data['Y']
    df['LeftEye_IsValid'] = data['IsValid']

    return df


def process_appleRightEye(user: Path):
    # H, W, X, Y, IsValid
    with open(user / 'appleRightEye.json') as json_file:
        data = json.load(json_file)

    df = pd.DataFrame()
    df['RightEye_H'] = data['H']
    df['RightEye_W'] = data['W']
    df['RightEye_X'] = data['X']
    df['RightEye_Y'] = data['Y']
    df['RightEye_IsValid'] = data['IsValid']

    return df


def process_dotInfo(user: Path):
    # DotNum, XPts, YPts, XCam, YCam, Time
    with open(user / 'dotInfo.json') as json_file:
        data = json.load(json_file)

    df = pd.DataFrame()
    df['dotInfo_DotNum'] = data['DotNum']
    df['dotInfo_XPts'] = data['XPts']
    df['dotInfo_YPts'] = data['YPts']
    df['dotInfo_XCam'] = data['XCam']
    df['dotInfo_YCam'] = data['YCam']
    df['dotInfo_Time'] = data['Time']

    return df


def process_faceGrid(user: Path):
    # H, W, X, Y, IsValid
    with open(user / 'faceGrid.json') as json_file:
        data = json.load(json_file)

    df = pd.DataFrame()
    df['faceGrid_H'] = data['H']
    df['faceGrid_W'] = data['W']
    df['faceGrid_X'] = data['X']
    df['faceGrid_Y'] = data['Y']
    df['faceGrid_IsValid'] = data['IsValid']

    return df


def process_frames(user: Path):
    # H, W, X, Y, IsValid
    with open(user / 'frames.json') as json_file:
        data = json.load(json_file)

    df = pd.DataFrame()
    df['frames'] = data

    return df


def process_info(user: Path, length: int):
    with open(user / 'info.json') as json_file:
        data = json.load(json_file)

    aux = apple_device_data[apple_device_data['DeviceName'] == data['DeviceName']]
    aux.reset_index(inplace=True, drop=True)
    aux.loc[0, 'TotalFrames'] = data['TotalFrames']
    aux.loc[0, 'NumFaceDetections'] = data['NumFaceDetections']
    aux.loc[0, 'NumEyeDetections'] = data['NumEyeDetections']
    aux.loc[0, 'Dataset'] = data['Dataset']

    df = pd.DataFrame()
    for l in range(length):
        df = df.append(aux, ignore_index=True)
    return df


def process_motion(user: Path, dot_info: pd.DataFrame):
    with open(user / 'motion.json') as json_file:
        data = json.load(json_file)

    df = pd.DataFrame(columns = ['motion_GravityX_Y', 'motion_GravityX_X', 'motion_GravityX_Z',
                  'motion_UserAcceleration_Y', 'motion_UserAcceleration_X', 'motion_UserAcceleration_Z',
                  'motion_AttitudeRotationMatrix_1', 'motion_AttitudeRotationMatrix_2', 'motion_AttitudeRotationMatrix_3',
                  'motion_AttitudeRotationMatrix_4', 'motion_AttitudeRotationMatrix_5', 'motion_AttitudeRotationMatrix_6',
                  'motion_AttitudeRotationMatrix_7', 'motion_AttitudeRotationMatrix_8', 'motion_AttitudeRotationMatrix_9',
                  'motion_AttitudePitch',
                  'motion_Time',
                  'motion_AttitudeQuaternion_X', 'motion_AttitudeQuaternion_W', 'motion_AttitudeQuaternion_Y', 'motion_AttitudeQuaternion_Z',
                  'motion_AttitudeRoll',
                  'motion_RotationRate_Y', 'motion_RotationRate_X', 'motion_RotationRate_Z',
                  'motion_AttitudeYaw',
                  'motion_DotNum'])

    # for d, i in zip(data, range(len(data))):
    data_index = 0
    data_length = len(data)
    p_m = False
    for i, frame in dot_info.iterrows():
        try:
            data_can = data[data_index]
            df.loc[i, 'motion_Correct'] = 1
        except:
            if not p_m:
                print('no available mobile data for: {}'.format(user.name))
                p_m = True
            df.loc[i, 'motion_Correct'] = 0
            # Now motion data and frames are align
            for j in ['Y', 'X', 'Z']:
                df.loc[i, 'motion_GravityX_'+j] = 0

            for j in ['Y', 'X', 'Z']:
                df.loc[i, 'motion_UserAcceleration_'+j] = 0

            for j in range(1,10):
                df.loc[i, 'motion_AttitudeRotationMatrix_'+str(j)] = 0

            df.loc[i, 'motion_AttitudePitch'] = 0
            df.loc[i, 'motion_Time'] = 0

            for j in ['X', 'W', 'Y', 'Z']:
                df.loc[i, 'motion_AttitudeQuaternion_'+j] = 0

            df.loc[i, 'motion_AttitudeRoll'] = 0

            for j in ['Y', 'X', 'Z']:
                df.loc[i, 'motion_RotationRate_'+j] = 0

            df.loc[i, 'motion_AttitudeYaw'] = 0
            df.loc[i, 'motion_DotNum'] = 0
            continue

        # We align motion data and frame data
        for d in range(data_index, data_length):
            data_this = data[d]
            if data_this['DotNum'] < frame['dotInfo_DotNum']:
                continue

            if data_this['Time'] > frame['dotInfo_Time']:
                data_index = d
                break
            else:
                data_can = data_this

        # Now motion data and frames are align
        for j in ['Y', 'X', 'Z']:
            df.loc[i, 'motion_GravityX_'+j] = data_can['GravityX'][j]

        for j in ['Y', 'X', 'Z']:
            df.loc[i, 'motion_UserAcceleration_'+j] = data_can['UserAcceleration'][j]

        for j in range(1,10):
            df.loc[i, 'motion_AttitudeRotationMatrix_'+str(j)] = data_can['AttitudeRotationMatrix'][j-1]

        df.loc[i, 'motion_AttitudePitch'] = data_can['AttitudePitch']
        df.loc[i, 'motion_Time'] = data_can['Time']

        for j in ['X', 'W', 'Y', 'Z']:
            df.loc[i, 'motion_AttitudeQuaternion_'+j] = data_can['AttitudeQuaternion'][j]

        df.loc[i, 'motion_AttitudeRoll'] = data_can['AttitudeRoll']

        for j in ['Y', 'X', 'Z']:
            df.loc[i, 'motion_RotationRate_'+j] = data_can['RotationRate'][j]

        df.loc[i, 'motion_AttitudeYaw'] = data_can['AttitudeYaw']
        df.loc[i, 'motion_DotNum'] = data_can['DotNum']

    return df


def process_screen(user: Path):
    # H, W, X, Y, IsValid
    with open(user / 'screen.json') as json_file:
        data = json.load(json_file)

    df = pd.DataFrame()
    df['Screen_H'] = data['H']
    df['Screen_W'] = data['W']
    df['Screen_Orientation'] = data['Orientation']

    return df


def process(user: Path, data_target: Path):
    df_fa = process_appleFace(user)
    df_le = process_appleLeftEye(user)
    df_re = process_appleRightEye(user)
    df_di = process_dotInfo(user)
    df_fg = process_faceGrid(user)
    df_fr = process_frames(user)
    df_in = process_info(user, len(df_fa))
    df_mo = process_motion(user, df_di)
    df_sc = process_screen(user)

    df_global = pd.concat([df_fa,
                           df_le,
                           df_re,
                           df_di,
                           df_fg,
                           df_fr,
                           df_in,
                           df_mo,
                           df_sc], axis=1)

    if not os.path.exists(data_target / user.name):
        os.mkdir(data_target / user.name)

    df_global.to_csv(data_target / user.name / 'table_info.csv')


class MyProcessor(Processor):

    def __init__(self):
        Processor.__init__(self)

    def process_task(self, task):
        user = task['user']
        data_target = task['data_target']

        process(user, data_target)

        return None

    def process_global_task(self):
        return 'done_' + self.name


if __name__ == '__main__':
    data_source = config.data_path / 'gaze'
    data_target = config.data_path / 'gaze_processing'
    parallel = False

    if not os.path.exists(data_target):
        os.mkdir(data_target)

    if parallel:
        i = 0
        for user in os.scandir(data_source):
            if user.is_dir():
                print(user)
                print(i)
                i += 1
                process(data_source / user, data_target)
    else:
        mpm = MultiProcessManager()

        for r in range(0, 8):
            mpm.add_processor(MyProcessor)

        sent = 0
        expl = 0

        real_exists = [d for d in os.listdir(data_target)]

        for user in os.scandir(data_source):
            if user.is_dir():
                expl += 1
                if not os.path.exists(data_target / user.name / 'table_info.csv'):
                    sent += 1
                    mpm.send_task({'user': data_source / user, 'data_target': data_target})
                else:
                    if not user.name in real_exists:
                        print('joder')

        received = 0
        while True:
            result = mpm.next_result()

            if result:
                print(result)
                received += 1
            else:
                break

        print('report')
        print('expl: {}'.format(expl))
        print('sent: {}'.format(sent))
        print('received: {}'.format(received))
        print('hola')