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
# e.g. if data_dir='../facescrub_all_cloaked_users/download/Portia_Doubleday/face/', returns 'Portia_Doubleday'
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
    def __init__(self, aligner=None):
        self.aligner = aligner
        select_data_dir = glob.glob(os.path.join(args.facescrub_dir, "*"))
        select_data_dir = sorted(select_data_dir)
        print("found {} directories".format(len(select_data_dir)))

        self.id2label = {}
        self.id2path = {}
        self.id2pathtest = {}
        idx = 0
        counts = [0, 0, 0, 0]
        dir1, dir2, dir3 = args.public_attack_dirs
        
        for cur_data_dir in select_data_dir:
            cur_class = get_class(cur_data_dir, face=False)
            
            if cur_class in args.names_list:
                print("IGNORING:", cur_data_dir)
                continue

            # UGLY: hardcoded filters for Fawkes v1.0, LowKey, Fawkes v0.3
            p1 = glob.glob(os.path.join(dir1 + cur_class + "/face/", "*high_cloaked.png"))
            p2 = glob.glob(os.path.join(dir2 + cur_class + "/face/", "*_attacked.png"))
            p3 = glob.glob(os.path.join(dir3 + cur_class + "/face/", "*high_cloaked.png"))

            if len(p1) + len(p2) + len(p3) == 0:
                continue
            
            self.id2label[cur_data_dir] = idx
            idx += 1

            all_pathes = glob.glob(os.path.join(cur_data_dir + "/face/", "*.jpg"))
            test_len = int(0.3 * len(all_pathes))
            test_path = random.sample(all_pathes, test_len)
            train_path = [p for p in all_pathes if p not in test_path]

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


class CallbackGenerator(keras.callbacks.Callback):
    def __init__(self, datagen, test_gen):
        self.datagen = datagen
        self.test_gen = test_gen

    def on_epoch_end(self, epoch, logs=None):
        _, other_acc = self.model.evaluate_generator(self.test_gen, verbose=0, steps=50)
        print("Epoch: {} - Other acc: {:.4f}".format(epoch, other_acc), flush=True)

        if epoch < 4 or (epoch+1) % 5 == 0:
            checkpoint_path = "cp-robust-{epoch:02d}.ckpt".format(epoch=epoch+1)
            self.model.save_weights(checkpoint_path, save_format='h5')


def main():
    sess = init_gpu(args.gpu)
    ali = aligner(sess)

    datagen = DataGenerator(aligner=ali)
    
    train_generator = datagen.generate()
    test_generator = datagen.generate(test=True)

    base_model = load_extractor(args.base_model)
    
    model = load_victim_model(teacher_model=base_model, number_classes=datagen.num_classes, end2end=True)

    cb = CallbackGenerator(datagen=datagen,
                           test_gen=test_generator)

    train_set_len = sum([len(datagen.id2path[i]) for i in datagen.id2label.keys() if i in datagen.id2path])
    print(f"len(train set) = {train_set_len}")

    model.fit_generator(train_generator, steps_per_epoch=500,
                        epochs=args.n_epochs,
                        verbose=1,
                        callbacks=[cb]
                        )

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')

    parser.add_argument('--base_model', type=str,
                        help='the feature extractor used for tracker model training. ', default='high_extract')
    parser.add_argument('--facescrub-dir', help='path to unprotected facescrub directory', default="facescrub/download/")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--public-attack-dirs', nargs='+', default=['facescrub_fawkesv10/download/', 'facescrub_lowkey/download/', 'facescrub_fawkesv03/download/'])
    parser.add_argument('--names-list', nargs='+', default=[], help="names of users for attack evaluation")

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
