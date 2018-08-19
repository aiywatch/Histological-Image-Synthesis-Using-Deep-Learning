# Put this file to the directory where 'Mask' and 'Sample' reside

import os
from shutil import copyfile

mask_names = next(os.walk('Mask'))[2]
image_names = next(os.walk('Sample'))[2]

NEW_ROOT_DIRECTORY = './MICCAI/'

if not os.path.exists(NEW_ROOT_DIRECTORY):
    os.makedirs(NEW_ROOT_DIRECTORY)

#print(image_names)
for image_name in image_names:
    print(image_name)
    name_parts = image_name.split('_')
    image_id = "{}_{}".format(name_parts[0], name_parts[2][:name_parts[2].index(".")])
    
    if not os.path.exists(NEW_ROOT_DIRECTORY + image_id):
        os.makedirs(NEW_ROOT_DIRECTORY + image_id)
        os.makedirs(NEW_ROOT_DIRECTORY + image_id + '/images')
        os.makedirs(NEW_ROOT_DIRECTORY + image_id + '/masks')

    mask_file_path = "./Mask/{}_mask_{}".format(name_parts[0], name_parts[2])
    image_save_path = '{}{}/images/{}'.format(NEW_ROOT_DIRECTORY, image_id, image_name)
    mask_save_path = '{}{}/masks/{}'.format(NEW_ROOT_DIRECTORY, image_id, image_name)
    
    copyfile('./Sample/'+image_name, image_save_path)
    copyfile(mask_file_path, mask_save_path)




