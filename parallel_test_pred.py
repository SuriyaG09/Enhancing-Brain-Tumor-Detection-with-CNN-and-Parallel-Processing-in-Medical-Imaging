import time
import multiprocessing
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('Braintumor_model.h5')

def process_images(image_paths):
    yes_count = 0  
    no_count = 0
    for path in image_paths:
        image = cv2.imread(path)
        img = Image.fromarray(image)
        img = img.resize((64,64))
        img = np.array(img)
        input_img = np.expand_dims(img, axis=0)
        result = model.predict(input_img)
        if result[0][0] == 1:
            yes_count += 1
            print('\nThe patient is suffering from a brain tumor')
        else:
            no_count += 1
            print('No tumor detected')
    return (yes_count, no_count)
start_time = time.time()
if __name__ == "__main__":
    num_processes =  multiprocessing.cpu_count()  # number of processes to use
    # paths = [f'datasets/yes/y{i}.jpg' for i in range(101, 301)]
    # no_paths = [f'datasets/no/no{i}.jpg' for i in range(101, 301)]
    paths = [f'pred/pred{i}.jpg' for i in range(0, 60)]
    pool = multiprocessing.Pool(processes=num_processes)
    results = []
    # no_results = []

    # divide images into equal chunks and process them in parallel
    chunk_size = len(paths) // num_processes
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size
        if i == num_processes - 1:
            end = len(paths)
        yes_slice = paths[start:end]
        # no_slice = no_paths[start:end]
        results.append(pool.apply_async(process_images, [yes_slice]))
        # no_results.append(pool.apply_async(process_images, [no_slice]))

    # get the results from each process and combine them
    total_yes_from_yes_image = 0
    total_no_from_yes_image = 0
    total_yes_from_no_image = 0
    total_no_from_no_image = 0
    for result in results:
        yes_count, no_count = result.get()
        total_yes_from_yes_image += yes_count
        total_no_from_yes_image += no_count
    # for result in no_results:
    #     yes_count, no_count = result.get()
    #     total_yes_from_no_image += yes_count
    #     total_no_from_no_image += no_count
    print("Testing of 400 images parallel:")
    print(f'Total Brain tumor suffered patients: {total_yes_from_yes_image}')
    print(f'Total Normal patients: {total_no_from_yes_image}')
    # print(f'Total yes from no image: {total_yes_from_no_image}')
    # print(f'Total no from no image: {total_no_from_no_image}')
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken:", total_time)