import argparse
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import random
import sys

import numpy as np
from keras.models import Model
from fawkes.align_face import aligner
from fawkes.utils import init_gpu, load_extractor, load_victim_model, preprocess, Faces, load_image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def filter_image_paths(image_paths):
    new_image_paths = []
    new_images = []
    for p in image_paths:
        img = load_image(p)
        if img is None:
            continue
        new_image_paths.append(p)
        new_images.append(img)
    return new_image_paths, new_images

def get_features(model, paths, ali, batch_size=16):
    paths, images = filter_image_paths(paths)
    faces = Faces(paths, images, ali, verbose=0, eval_local=True, no_align=True)
    faces = faces.cropped_faces
    features = model.predict(faces, verbose=0)
    return features

def get_feature_extractor(base_model="low_extract", custom_weights=None):
    base_model = load_extractor(base_model)
    features = base_model.layers[-1].output
    model = Model(inputs=base_model.input, outputs=features)

    if custom_weights is not None:
        model.load_weights(custom_weights, by_name=True, skip_mismatch=True)

    return model

def get_class(data_dir):
    folders_arr = data_dir.split('/')
    for i in range(len(folders_arr)-1):
        if folders_arr[i+1] == 'face':
            class_name = folders_arr[i]
            return class_name
    return None

def get_facescrub_features(model, ali, dataset_path):
    # get features for all facescrub users
    data_dirs = sorted(glob.glob(os.path.join(dataset_path, "*")))

    classes_train = []
    features_train = []

    classes_test = []
    features_test = []

    for data_dir in data_dirs:
        data_dir += "/face/"
        cls = get_class(data_dir)
        all_pathes = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))

        f = get_features(model, all_pathes, ali)

        test_len = int(0.3 * len(all_pathes))
        test_idx = random.sample(range(len(all_pathes)), test_len)

        f_test = f[test_idx]
        f_train = np.delete(f, test_idx, axis=0)
        features_train.append(f_train)
        classes_train.extend([cls] * len(f_train))
        features_test.append(f_test)
        classes_test.extend([cls] * len(f_test))

    classes_train = np.asarray(classes_train)
    features_train = np.concatenate(features_train, axis=0)

    classes_test = np.asarray(classes_test)
    features_test = np.concatenate(features_test, axis=0)

    return features_train, features_test, classes_train, classes_test

def main():
    sess = init_gpu("0")
    ali = aligner(sess)
    model = get_feature_extractor("low_extract", custom_weights=args.robust_weights)

    random.seed(10)
    print("Extracting features...", flush=True)
    X_train_all, X_test_all, Y_train_all, Y_test_all = get_facescrub_features(model, ali, args.facescrub_dir)

    val_people = args.names_list
    print(val_people)

    base_dir = args.attack_dir

    for name in val_people:
        directory = f"{base_dir}/{name}/face/"
        print(directory)
        image_paths = glob.glob(directory + "*.png") + glob.glob(directory + "*.jpg")
        
        """
        if args.attack == "fawkes" or args.attack == "fawkesv01":
            cloak_file_name = "high_cloaked"
            all_pathes_uncloaked = sorted([path for path in image_paths if "cloaked" not in path.split("/")[-1]])
            all_pathes_cloaked = sorted([path for path in image_paths if cloak_file_name in path.split("/")[-1]])    
        else:
            cloak_file_name = "attacked"
            all_pathes_uncloaked = sorted([path for path in image_paths if "small" in path.split("/")[-1]])
            all_pathes_cloaked = sorted([path for path in image_paths if cloak_file_name in path.split("/")[-1]])    
        """

        all_pathes_uncloaked = sorted([path for path in image_paths if args.unprotected_file_match in path.split("/")[-1]])
        all_pathes_cloaked = sorted([path for path in image_paths if args.protected_file_match in path.split("/")[-1]])  

        print(name, len(all_pathes_cloaked), len(all_pathes_uncloaked))
        assert len(all_pathes_cloaked) == len(all_pathes_uncloaked)
        
        f_cloaked = get_features(model, all_pathes_cloaked, ali)
        f_uncloaked = get_features(model, all_pathes_uncloaked, ali)

        random.seed(10)
        test_frac = 0.3
        test_idx = random.sample(range(len(all_pathes_cloaked)), int(test_frac * len(all_pathes_cloaked)))

        f_train_cloaked = np.delete(f_cloaked, test_idx, axis=0)
        f_test_cloaked = f_cloaked[test_idx]
 
        f_train_uncloaked = np.delete(f_uncloaked, test_idx, axis=0)
        f_test_uncloaked = f_uncloaked[test_idx]

        if args.classifier == "linear":
            clf1 = LogisticRegression(random_state=0, n_jobs=-1, warm_start=False) 
            clf1 = make_pipeline(StandardScaler(), clf1)
        else:
            clf1 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)

        idx_train = np.asarray([y != name for y in Y_train_all])
        idx_test = np.asarray([y != name for y in Y_test_all])
        print(np.sum(idx_train), np.sum(idx_test))

        # with cloaking
        X_train = np.concatenate((X_train_all[idx_train], f_train_cloaked))
        Y_train = np.concatenate((Y_train_all[idx_train], [name] * len(f_train_cloaked)))
        clf1 = clf1.fit(X_train, Y_train)

        print("Test acc: {:.2f}".format(clf1.score(X_test_all[idx_test], Y_test_all[idx_test])))
        print("Train acc (user cloaked): {:.2f}".format(clf1.score(f_train_cloaked, [name] * len(f_train_cloaked))))
        print("Test acc (user cloaked): {:.2f}".format(clf1.score(f_test_uncloaked, [name] * len(f_test_uncloaked))))
        print("Protection rate: {:.2f}".format(1-clf1.score(f_test_uncloaked, [name] * len(f_test_uncloaked))))
        print(flush=True)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_model', type=str,
                        help='the feature extractor', default='low_extract')
    parser.add_argument('--classifier', type=str,
                        help='the classifier', default='linear')
    parser.add_argument('--robust-weights', type=str, 
                        help='robust weights', default=None)
    parser.add_argument('--names-list', nargs='+', default=[], help="names of attacking users")
    parser.add_argument('--facescrub-dir', help='path to unprotected facescrub directory', default="facescrub/download/")
    parser.add_argument('--attack-dir', help='path to protected facescrub directory', default="facescrub_attacked/download/")
    parser.add_argument('--unprotected-file-match', type=str,
                        help='pattern to match protected pictures', default='.jpg')
    parser.add_argument('--protected-file-match', type=str,
                        help='pattern to match protected pictures', default='high_cloaked.png')
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
