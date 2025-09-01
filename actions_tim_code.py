import numpy as np

def copy_action_from_csv(csv, row):
    """
    Creates numpy array of actions from CSV row

    Parameters:
        | csv (np.ndarray): object
        | row (int): index

    Returns:
        | np.ndarray numpy array of actions
    """
    cols = csv.shape[1] - 2
    new_action = np.zeros((1, cols), dtype=np.uint8)
    for i in range(cols):

        # check if we load any nan values from config and set them to 0 (currently touchpad_id is nan)
        value = csv[row, 2 + i]
        if not np.isnan(value):
            new_action[0, i] = value.astype(np.uint8)
        else:
            new_action[0, i] = 0

    return new_action

# obviously just part of a larger function and will need adjusting
        # output actions array
        actions_csv = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
        # -2 to remove index and timestamp coloumns
        actions_shape = (num_records, actions_csv.shape[1] - 2)
        actions_buffer = np.zeros(actions_shape)
        current_action = 0
        action_timestamp = int(actions_csv[current_action, 1])
        ms_per_image = (1.0 / video_fps) * 1000.0

        for x in range(actions_shape[0]):
            # print(
            #    "Output actions numpy: Adding state #"
            #    + str(x)
            #    + " of "
            #    + str(actions_shape[0]),
            #    end="\r",
            #    flush=True,
            # )

            # match ctrlp status with each state image
            img_timestamp = int(x * ms_per_image)

            # make sure we're using the most recent (or equal) ctrlp state to the image state
            # keep iterating actions to find the first action after the current state
            while action_timestamp < img_timestamp:
                current_action += 1
                if current_action >= actions_csv.shape[0]:
                    # reached the last action, so repeat it
                    break
                action_timestamp = int(actions_csv[current_action, 1])
            # then back up one (if not 0)
            if current_action > 0:
                current_action -= 1
                action_timestamp = int(actions_csv[current_action, 1])

            # copy into actions numpy buffer (numpy states buffer forms a (state,action)
            # pair with this actions buffer)
            actions_buffer[x] = copy_action_from_csv(actions_csv, current_action)