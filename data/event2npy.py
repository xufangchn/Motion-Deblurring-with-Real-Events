import numpy as np
import os
import copy

path = './HQF'

image_height = 180
image_width = 240

bag_list_file = open(os.path.join(path, 'test_bags.txt'), 'r')
lines = bag_list_file.read().splitlines()
bag_list_file.close()

for i in range(len(lines)):
    seq = lines[i]
    print(seq)
    event_txt = os.path.join(path, seq, 'events.txt')
    seq_save_path = os.path.join(path, seq, 'npy')
    if not os.path.exists(seq_save_path):
        os.makedirs(seq_save_path)

    f = open(event_txt)
    lines = f.readlines()
    event_timestamp = []
    event_x = []
    event_y = []
    event_polarity = []
    for line in lines:
        list = line.strip('\n').split(' ')
        event_timestamp.append(float(list[0]))
        event_x.append(int(list[1]))
        event_y.append(int(list[2]))
        event_polarity.append(int(list[3]))
    f.close()
    event_timestamp = np.array(event_timestamp)
    event_x = np.array(event_x)
    event_y = np.array(event_y)
    event_polarity = np.array(event_polarity)

    image_txt = os.path.join(path, seq, 'images.txt')
    f = open(image_txt)
    lines = f.readlines()
    frame_timestamp = []
    for line in lines:
        list = line.strip('\n').split(' ')
        frame_timestamp.append(float(list[0]))
    f.close()
    frame_timestamp = np.array(frame_timestamp)

    ids = np.searchsorted(event_timestamp, frame_timestamp, side='right')
    ids = ids.astype(int)

    _MAX_SKIP_FRAMES = 1
    event_count_images = []
    event_time_images = []

    for h in range(len(frame_timestamp)):
        if h > 0:
            event_count_image = np.zeros([1,image_height,image_width,2],dtype=np.uint16)
            event_time_image = np.zeros([1,image_height,image_width,2],dtype=np.float64)

            start_event_id = ids[h-1]
            end_event_id = ids[h]-1
            # section event
            section_event_timestamp = copy.deepcopy(event_timestamp[start_event_id:end_event_id+1])
            section_event_x = copy.deepcopy(event_x[start_event_id:end_event_id+1])
            section_event_y = copy.deepcopy(event_y[start_event_id:end_event_id+1])
            section_event_polarity = copy.deepcopy(event_polarity[start_event_id:end_event_id+1])
            # section pos event
            section_pos_event_timestamp = section_event_timestamp[section_event_polarity==1]
            section_pos_event_x = section_event_x[section_event_polarity==1]
            section_pos_event_y = section_event_y[section_event_polarity==1]
            # section neg event
            section_neg_event_timestamp = section_event_timestamp[section_event_polarity<1]
            section_neg_event_x = section_event_x[section_event_polarity<1]
            section_neg_event_y = section_event_y[section_event_polarity<1]

            section_event_num = end_event_id - start_event_id + 1
            print(h, '/', len(frame_timestamp), ',', section_event_num)

            event_count_image_pos = np.zeros(image_height*image_width, dtype=np.uint16)
            section_pos_event_x = section_pos_event_x.astype(int)
            section_pos_event_y = section_pos_event_y.astype(int)
            section_index_pos = section_pos_event_y*image_width+section_pos_event_x
            np.add.at(event_count_image_pos, section_index_pos, 1)
            event_count_image_pos = event_count_image_pos.reshape([image_height, image_width])

            event_count_image_neg = np.zeros(image_height*image_width, dtype=np.uint16)
            section_neg_event_x = section_neg_event_x.astype(int)
            section_neg_event_y = section_neg_event_y.astype(int)
            section_index_neg = section_neg_event_y*image_width+section_neg_event_x
            np.add.at(event_count_image_neg, section_index_neg, 1)
            event_count_image_neg = event_count_image_neg.reshape([image_height, image_width])

            event_count_image[0,:,:,0] = event_count_image_pos
            event_count_image[0,:,:,1] = event_count_image_neg

            event_time_image[0,section_pos_event_y,section_pos_event_x,0] = section_pos_event_timestamp
            event_time_image[0,section_neg_event_y,section_neg_event_x,1] = section_neg_event_timestamp

            if h==1:
                event_count_images = event_count_image
                event_time_images = event_time_image
            else:
                event_count_images = np.concatenate([event_count_images, event_count_image],axis=0)
                event_time_images = np.concatenate([event_time_images, event_time_image],axis=0)

            if h>=_MAX_SKIP_FRAMES:
                out_event_count_images = copy.deepcopy(event_count_images)
                out_event_time_images = copy.deepcopy(event_time_images) 
                out_event_time_images[out_event_time_images>0] -= frame_timestamp[h-_MAX_SKIP_FRAMES]
                Fout_event_time_images = np.zeros((_MAX_SKIP_FRAMES, image_height, image_width, 2), dtype=np.float32)
                Fout_event_time_images = copy.deepcopy(out_event_time_images)
                out_event_images = np.array([np.array(out_event_count_images),np.array(Fout_event_time_images)])
                np.save(os.path.join(seq_save_path, "event{:05d}".format(h-_MAX_SKIP_FRAMES)), out_event_images)
                event_count_images = np.delete(event_count_images, 0, axis=0)
                event_time_images = np.delete(event_time_images, 0, axis=0)
            











    

    

        
    