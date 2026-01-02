import os
import random


class GetRandomData:
    def __init__(self, lr_filepath, hr_filepath, n_train=0, n_validation=0, is_random=True):
        self.lr_filepath = lr_filepath
        self.hr_filepath = hr_filepath
        self.n_train = n_train
        self.n_validation = n_validation

        self.is_random = is_random

    def get_data(self):
        # list all data
        lr_files = os.listdir(self.lr_filepath)
        hr_files = os.listdir(self.hr_filepath)

        if self.is_random:
            random.seed(24)
            random.shuffle(lr_files)

        if self.n_train == 0:
            self.n_train = len(lr_files)
            self.n_validation = 0

        train_files = lr_files[:self.n_train]

        print(f"Training data: {len(train_files)}")

        # Select corresponding HR samples
        hr_lookup = {os.path.basename(f).split('_')[0]: f for f in hr_files}
        hr_train_files = []

        for lr_path in train_files:
            lr_id = os.path.basename(lr_path).split('_')[0]
            if lr_id in hr_lookup:
                hr_train_files.append(hr_lookup[lr_id])
            else:
                print(f"Couldn't find HR file for '{lr_id}'")

        validation_files = []
        hr_validation_files = []
        if self.n_validation != 0:
            validation_files = lr_files[self.n_train: self.n_train + self.n_validation]
            print(f"Validation data: {len(validation_files)}")
            hr_validation_files = [
                hr_lookup[os.path.basename(f).split('_')[0]]
                for f in validation_files
                if os.path.basename(f).split('_')[0] in hr_lookup
            ]

        return train_files, validation_files, hr_train_files, hr_validation_files

