import argparse
import collections
import deepmvlm
import os
from parse_config import ConfigParser
from utils import mesh_files_in_dir, read_landmarks, read_lines, save_excel_metadata
from utils3d import *

def processOneFile(config, file_name):
    # Get basename
    basename = os.path.basename(file_name).split('.')[0]

    # Predict landmarks
    dm = deepmvlm.DeepMVLM(config)
    landmarks = dm.predict(file_name, basename, config.output_path / '_results_ko.txt',config.output_path)[0]

    # Save Excel metadata
    if config.metadata_save:
        if landmarks is not None:
            metadata_filename = str(config.output_path / 'metadata_landmarks.xlsx')
            save_excel_metadata(metadata_filename,basename,landmarks)

    # Show prediction
    lm_num = config[m.arch][m.arch_args][m.arch_args_nlm]
    if landmarks is not None:
        name_lm = os.path.splitext(file_name)[0] + '.txt'
        origLandmarks = read_landmarks(name_lm)
        png_file = str(config.temp_dir / (basename + '.png'))
        if config.save_img:
            print('Storing', basename + '.png', 'on', str(config.temp_dir))
            visualise_mesh_and_landmarks(file_name, landmarks, origLandmarks[0:lm_num] if origLandmarks is not None else None,
                off_screen=True, out_file=png_file)
        if config.render_predict:
            visualise_mesh_and_landmarks(file_name, landmarks, origLandmarks[0:lm_num] if origLandmarks is not None else None)


def processFileList(config, file_name):
    # Get list of files
    print('Processing filelist ', file_name)
    names = read_lines(file_name)

    # Prediction
    lm_num = config[m.arch][m.arch_args][m.arch_args_nlm]
    print('Processing ', len(names), ' meshes')
    dm = deepmvlm.DeepMVLM(config)
    for file_name in names:
        # Predict landmarks
        basename = os.path.basename(file_name).split('.')[0]
        landmarks = dm.predict(file_name, basename, config.output_path / '_results_ko.txt',config.output_path)[0]

        # Save Excel metadata
        if config.metadata_save:
            if landmarks is not None:
                metadata_filename = str(config.output_path / 'metadata_landmarks.xlsx')
                save_excel_metadata(metadata_filename,basename,landmarks)

        # Show prediction
        if landmarks is not None:
            name_lm = os.path.splitext(file_name)[0] + '.txt'
            origLandmarks = read_landmarks(name_lm)
            png_file = str(config.temp_dir / (basename + '.png'))
            if config.save_img:
                print('Storing', basename + '.png', 'on', str(config.temp_dir))
                visualise_mesh_and_landmarks(file_name, landmarks, origLandmarks[0:lm_num] if origLandmarks is not None else None,
                    off_screen=True, out_file=png_file)
            if config.render_predict:
                visualise_mesh_and_landmarks(file_name, landmarks, origLandmarks[0:lm_num] if origLandmarks is not None else None)


def processDirFiles(config, dir_name):
    # Get list of files in directory
    print('Processing files in  ', dir_name)
    names = mesh_files_in_dir(dir_name)

    # Prediction
    lm_num = config[m.arch][m.arch_args][m.arch_args_nlm]
    print('Processing ', len(names), ' meshes')
    dm = deepmvlm.DeepMVLM(config)
    for file_name in names:
        # Predict landmarks
        basename = os.path.basename(file_name).split('.')[0]
        landmarks = dm.predict(file_name, basename, config.output_path / '_results_ko.txt', config.output_path)[0]

        # Save Excel metadata
        if config.metadata_save:
            if landmarks is not None:
                metadata_filename = str(config.output_path / 'metadata_landmarks.xlsx')
                save_excel_metadata(metadata_filename,basename,landmarks)

        # Show prediction
        if landmarks is not None:
            name_lm = os.path.splitext(file_name)[0] + '.txt'
            origLandmarks = read_landmarks(name_lm)
            png_file = str(config.temp_dir / (basename + '.png'))
            if config.save_img:
                print('Storing', basename + '.png', 'on', str(config.temp_dir))
                visualise_mesh_and_landmarks(file_name, landmarks, origLandmarks[0:lm_num] if origLandmarks is not None else None,
                    off_screen=True, out_file=png_file)
            if config.render_predict:
                visualise_mesh_and_landmarks(file_name, landmarks, origLandmarks[0:lm_num] if origLandmarks is not None else None)


def main(config):
    name = str(config.name)
    if name.lower().endswith(('.obj', '.wrl', '.vtk', '.ply', '.stl')) and os.path.isfile(name):
        processOneFile(config, name)
    elif name.lower().endswith('.txt') and os.path.isfile(name):
        processFileList(config, name)
    elif os.path.isdir(name):
        processDirFiles(config, name)
    else:
        print('Cannot process (not a mesh file, a filelist (.txt) or a directory)', name)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Deep-MVLM')
    args.add_argument('-c', '--config', default=None, type=str,
        help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
        help='indices of GPUs to enable (default: all)')
    args.add_argument('-n', '--name', default=None, type=str,
        help='name of file, filelist (.txt) or directory to be processed')
    args.add_argument('-pn', '--predict_num', default=None, type=str,
        help='total predictions to average per model (default: 10)')
    args.add_argument('-pt', '--predict_tries', default=None, type=str,
        help='total prediction tries before dismiss (default: 3)')
    args.add_argument('-mr', '--max_ransac', default=None, type=str,
        help='Ransac error maximum tolerance (default: 5)')
    args.add_argument('-rp', '--render_predict', default=None, type=str,
        help='Render predictions (default: false)')
    args.add_argument('-si', '--save_img', default=None, type=str,
        help='Save prediction as 2D image (default: false)')
    args.add_argument('-o', '--output_path', default=None, type=str,
        help='Set output path (default: None)')
    args.add_argument('-of', '--output_format', default=None, type=str,
        help='Set output format (default: txt)')
    args.add_argument('-ms', '--metadata_save', default=None, type=str,
        help='Save Excel metadata (default: false)')

    # Custom cli options to modify configuration from default values given in config file
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-ng', '--n_gpu'], type=int, target=[m.ngpu])
    ]
    global_config = ConfigParser(args, options)
    main(global_config)
