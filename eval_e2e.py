import argparse
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import random
random.seed(10)

import sys

import keras
import numpy as np
from fawkes.align_face import aligner
from fawkes.utils import init_gpu, load_extractor, load_victim_model, preprocess, Faces, filter_image_paths
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.layers import Dense, Activation
from keras.models import Model

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def horizontal_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, ::-1, :]
    return image

def select_samples(data_dir):
    all_data_path = []
    for cls in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls)
        for data_path in os.listdir(cls_dir):
            all_data_path.append(os.path.join(cls_dir, data_path))
    return all_data_path

# returns the name of the class from the face directory
# e.g. if data_dir='/facescrub/download/Portia_Doubleday/face/', returns 'Portia_Doubleday'
def get_class(data_dir, face=True):
    folders_arr = data_dir.split('/')
    if not face:
        return folders_arr[-1]
    for i in range(len(folders_arr)-1):
        if folders_arr[i+1] == 'face':
            class_name = folders_arr[i]
            return class_name
    return None
    

class DataGenerator(object):
    def __init__(self, original_images, protect_images, cloak_frac=1.0, aligner=None):
        l = int(len(original_images) * 0.7)
        cloak_l = int(cloak_frac*l)
        self.original_images_train = original_images[cloak_l:l]
        self.original_images_test = original_images[l:]
        self.protect_images_train = np.concatenate(
            (protect_images[:cloak_l], self.original_images_train), axis=0)
        self.protect_images_test = protect_images[l:]

        print("# of total train images:", len(self.protect_images_train))
        
        self.aligner = aligner
        select_data_dir = glob.glob(os.path.join(args.facescrub_dir, "*"))
        select_data_dir = sorted(select_data_dir)
        print("found {} directories".format(len(select_data_dir)))

        self.id2label = {"-1": 0}
        self.id2path = {}
        self.id2pathtest = {}
        idx = 1
        
        counts = [0, 0, 0, 0]
        
        for cur_data_dir in select_data_dir:
            cur_class = get_class(cur_data_dir, face=False)
            
            if get_class(args.attack_dir) in cur_data_dir:
                print("IGNORING:", cur_data_dir)
                continue

            self.id2label[cur_data_dir] = idx
            idx += 1

            all_pathes = glob.glob(os.path.join(cur_data_dir + "/face/", "*.jpg"))
            test_len = int(0.3 * len(all_pathes))
            test_path = random.sample(all_pathes, test_len)
            train_path = [p for p in all_pathes if p not in test_path]

            if args.robust:
                # UGLY: hardcoded filters for Fawkes v1.0, LowKey, Fawkes v0.3
                dir1, dir2, dir3 = args.public_attack_dirs
                p1 = glob.glob(os.path.join(dir1 + cur_class + "/face/", "*high_cloaked.png"))
                p2 = glob.glob(os.path.join(dir2 + cur_class + "/face/", "*_attacked.png"))
                p3 = glob.glob(os.path.join(dir3 + cur_class + "/face/", "*high_cloaked.png"))

                if len(p1) + len(p2) + len(p3) > 0:
                    path1 = []
                    path2 = []
                    path3 = []

                    for p in train_path:
                        fname = os.path.splitext(os.path.basename(p))[0]
                        fname = fname.split("_")[0]

                        l1 = [p for p in p1 if fname in p]
                        l2 = [p for p in p2 if fname in p]
                        l3 = [p for p in p3 if fname in p]

                        if l1:
                            path1.append(l1[0])
                        if l2:
                            path2.append(l2[0])
                        if l3:
                            path3.append(l3[0])
                    
                    if len(path1):
                        counts[0] += 1
                    if len(path2):
                        counts[1] += 1
                    if len(path3):
                        counts[2] += 1
                    if len(path1) & len(path2) & len(path3):
                        counts[3] += 1

                    train_path.extend(path1)
                    train_path.extend(path2)
                    train_path.extend(path3)
            
            self.id2path[cur_data_dir] = train_path
            self.id2pathtest[cur_data_dir] = test_path

        self.all_id = list(self.id2label.keys())
        self.num_classes = len(self.all_id)
        print("num classes: {}, num ids: {}".format(self.num_classes, len(self.all_id)))
        print(counts)

    def generate(self, test=False):
        while True:
            batch_X = []
            batch_Y = []

            cur_batch_path = np.random.choice(self.all_id, args.batch_size)
            for p in cur_batch_path:
                cur_y = self.id2label[p]
                if test and p == '-1':
                    continue
                # protect class images in train dataset
                elif p == '-1':
                    cur_x = random.choice(self.protect_images_train)
                else:
                    if test:
                        cur_path = random.choice(self.id2pathtest[p])
                    else:
                        cur_path = random.choice(self.id2path[p])

                    im = image.load_img(cur_path)
                    cur_x = image.img_to_array(im)
                    faces = Faces([im], [cur_x], self.aligner, verbose=0, eval_local=True, no_align=True)
                    cur_x = faces.cropped_faces[0]
                
                if not test:
                    if np.random.randint(2):
                        cur_x = horizontal_flip(cur_x)

                if cur_x is not None:
                    batch_X.append(cur_x)
                    batch_Y.append(cur_y)
                    
            batch_X = np.array(batch_X)
            batch_Y = to_categorical(np.array(batch_Y), num_classes=self.num_classes)

            yield batch_X, batch_Y


    def test_original(self):
        original_y = to_categorical([0] * len(self.original_images_test), num_classes=self.num_classes)
        return self.original_images_test, original_y

    def test_cloaked(self):
        original_y = to_categorical([0] * len(self.protect_images_test), num_classes=self.num_classes)
        return self.protect_images_test, original_y

    def train_cloaked(self):
        original_y = to_categorical([0] * len(self.protect_images_train), num_classes=self.num_classes)
        return self.protect_images_train, original_y



class CallbackGenerator(keras.callbacks.Callback):
    def __init__(self, original_imgs, protect_imgs, original_y, original_protect_y, datagen, test_gen):
        self.original_imgs = original_imgs
        self.protect_imgs = protect_imgs

        self.original_y = original_y
        self.original_protect_y = original_protect_y
        self.datagen = datagen
        self.test_gen = test_gen

    def on_epoch_end(self, epoch, logs=None):
        cloak_train_X, original_train_Y = self.datagen.train_cloaked()
        _, acc_cloak = self.model.evaluate(cloak_train_X, original_train_Y, verbose=0)
        print("\nEpoch: {} - Train acc on cloaked: {:.4f}".format(epoch, acc_cloak), flush=True)

        cloak_test_X, original_test_Y = self.datagen.test_cloaked()
        _, acc_cloak = self.model.evaluate(cloak_test_X, original_test_Y, verbose=0)
        print("Epoch: {} - Test acc on cloaked: {:.4f}".format(epoch, acc_cloak), flush=True)

        _, original_acc = self.model.evaluate(self.original_imgs, self.original_y, verbose=0)
        print("Epoch: {} - Protection success rate: {:.4f}".format(epoch, 1 - original_acc), flush=True)

        _, other_acc = self.model.evaluate_generator(self.test_gen, verbose=0, steps=50)
        print("Epoch: {} - Other acc: {:.4f}".format(epoch, other_acc), flush=True)


def main():
    sess = init_gpu(args.gpu)
    ali = aligner(sess)
    image_paths = glob.glob(os.path.join(args.attack_dir, "*"))

    original_image_paths = sorted([path for path in image_paths if args.unprotected_file_match in path.split("/")[-1]])
    original_image_paths, original_loaded_images = filter_image_paths(original_image_paths)

    protect_image_paths = sorted([path for path in image_paths if args.protected_file_match in path.split("/")[-1]])
    protect_image_paths, protected_loaded_images = filter_image_paths(protect_image_paths)

    print("Find {} original image and {} cloaked images".format(len(original_image_paths), len(protect_image_paths)))

    original_faces = Faces(original_image_paths, original_loaded_images, ali, verbose=0, eval_local=True, no_align=True)
    original_faces = original_faces.cropped_faces
    cloaked_faces = Faces(protect_image_paths, protected_loaded_images, ali, verbose=0, eval_local=True, no_align=True)
    cloaked_faces = cloaked_faces.cropped_faces

    if len(original_faces) <= 10 or len(protect_image_paths) <= 10:
        raise Exception("Must have more than 10 protected images to run the evaluation")

    datagen = DataGenerator(original_faces, cloaked_faces, aligner=ali)
    original_test_X, original_test_Y = datagen.test_original()
    print("{} Training Images | {} Testing Images".format(len(datagen.protect_images_train), len(original_test_X)), flush=True)
    
    train_generator = datagen.generate()
    test_generator = datagen.generate(test=True)

    train_set_len = sum([len(datagen.id2path[i]) for i in datagen.id2label.keys() if i in datagen.id2path])
    print(f"len(train set) = {train_set_len}", flush=True)

    base_model = load_extractor(args.base_model)
    
    model = load_victim_model(teacher_model=base_model, number_classes=datagen.num_classes, end2end=True)

    cb = CallbackGenerator(original_imgs=original_test_X, protect_imgs=cloaked_faces, original_y=original_test_Y,
                           original_protect_y=None,
                           datagen=datagen,
                           test_gen=test_generator)

    model.fit_generator(train_generator, steps_per_epoch=500,
                        epochs=args.n_epochs,
                        verbose=args.verbose,
                        callbacks=[cb]
                        )

    cloak_train_X, original_test_Y = datagen.train_cloaked()
    _, acc_cloak = model.evaluate(cloak_train_X, original_test_Y, verbose=0)
    print("\nTrain acc on cloaked: {:.4f}".format(acc_cloak))

    cloak_test_X, original_test_Y = datagen.test_cloaked()
    _, acc_cloak = model.evaluate(cloak_test_X, original_test_Y, verbose=0)
    print("Test acc on cloaked: {:.4f}".format(acc_cloak))

    original_test_X, original_test_Y = datagen.test_original()
    _, acc_original = model.evaluate(original_test_X, original_test_Y, verbose=0)
    print("Protection success rate: {:.4f}".format(1 - acc_original), flush=True)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')

    parser.add_argument('--base_model', type=str,
                        help='the feature extractor used for tracker model training. ', default='low_extract')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)

    parser.add_argument('--facescrub-dir', help='path to unprotected facescrub directory (all users)', default="facescrub/download/")
    parser.add_argument('--attack-dir', help='path to protected facescrub directory (single user)', default="facescrub/download/Aaron_Eckhart/face/")
    parser.add_argument('--unprotected-file-match', type=str,
                        help='pattern to match protected pictures', default='.jpg')
    parser.add_argument('--protected-file-match', type=str,
                        help='pattern to match protected pictures', default='high_cloaked.png')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--robust', action='store_true')
    parser.add_argument('--public-attack-dirs', nargs='+', default=['facescrub_fawkesv10/download/', 'facescrub_lowkey/download/', 'facescrub_fawkesv03/download/'])

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
