import numpy as np

    def build_lookup_table(self):
        self.lookup_table = {}

        starting_index = 0
        self.cumulative_sequences = 0
        # loop through all trajectories
        for trajectory_id in self.trajectory_id_list:
            actions_path = (
                self.root_dir + "/" + str(trajectory_id) + "/numpy/actions.npy"
            )
            src_actions = np.load(actions_path, allow_pickle=True)
            # do these ever not line up?
            num_records = src_actions.item().get("actions").shape[0]
            num_sequences = len(range(self.start_record, num_records, self.record_step))

            self.cumulative_sequences += num_sequences
            self.lookup_table[self.cumulative_sequences] = [
                trajectory_id,
                num_records,
                num_sequences,
                starting_index,
            ]
            starting_index = self.cumulative_sequences

    def __getitem__(self, idx):
        lookup_key = None
        for i in self.lookup_table.keys():
            if i > idx:
                lookup_key = i
                break

        lookup_values = self.lookup_table[lookup_key]

        trajectory_id = lookup_values[0]
        num_records = lookup_values[1]
        num_sequences = lookup_values[2]
        starting_index = lookup_values[3]

        numpy_path = self.root_dir + "/" + str(trajectory_id) + "/numpy/"