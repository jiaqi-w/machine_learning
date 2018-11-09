#Created by Jiaqi, Jan 19, 2017
#Updated by Jiaqi, March 14, 2017

import csv, collections, os
import numpy

def get_data(in_dir, label_fields, worker_id_field, text_repeat_times):
    worker_ids = set()
    text_worker_label = collections.defaultdict(dict)
    if os.path.isdir(in_dir):
        for f in os.listdir(in_dir):
            fname = os.path.join(in_dir, f)
            update_dict(fname, label_fields, worker_id_field, text_repeat_times, worker_ids, text_worker_label)
    else:
        fname = in_dir
        update_dict(fname, label_fields, worker_id_field, text_repeat_times, worker_ids, text_worker_label)

    text_ids = text_worker_label.keys()
    all_labels = []
    for worker_id in worker_ids:
        worker_labels = []
        for text_id in text_ids:
            if worker_id in text_worker_label[text_id]:
                worker_labels.append(text_worker_label[text_id][worker_id])
            else:
                worker_labels.append("*")
        all_labels.append(worker_labels)
    print("num of workers ", len(all_labels))
    print("num of items ", len(all_labels[0]))
    return numpy.array(all_labels)

def update_dict(in_fname, label_fields, worker_id_field, text_repeat_times, worker_ids, text_worker_label):
    with open(in_fname) as in_file:
        reader = csv.DictReader(in_file)
        print("Read " + in_fname)
        turk_count = 0
        label_boundary = len(label_fields)
        base_count = 0
        base_name = os.path.basename(in_fname)
        for row in reader:
            turk_count += 1
            work_id = row[worker_id_field]
            worker_ids.add(work_id)
            # i is the index of questions.
            for i in range(0, label_boundary):
                # field_label can be "positive", "negative", "neither"
                field_label = row[label_fields[i]]
                field_value = "0"
                if field_label == "Fulfilled":
                    field_value = "1"
                elif field_label == "Unfulfilled":
                    field_value = "2"
                elif field_label == "Unknown from the context":
                    field_value = "3"

                text_id = "{}_{}".format(base_name, str(base_count + i ))
                text_worker_label[text_id][work_id] = field_value

            # When collect all the labels for the same text.
            if turk_count == text_repeat_times:
                # reset
                turk_count = 0
                base_count += label_boundary
        print("Finish converting.")

def get_label_names(label_name_format, max_index, min_index=1):
    label_names = []
    for i in range(min_index, max_index + 1):
        lname = label_name_format.format(str(i))
        label_names.append(lname)
    return label_names

if __name__ == "__main__":
    num_annotate_each_line = 5
    label_fields = get_label_names("Answer.P{}Q2Answer", num_annotate_each_line)
    worker_id_field='WorkerId'
    num_annotate_each_text = 3  # repeaet 3 times
    get_data("./data", label_fields, worker_id_field, num_annotate_each_text)

