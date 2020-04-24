from collections import defaultdict



class star_ejecta(object):
    complete_values = {"SNII": 81,
                       "SNIa": 77,
                       "wind": 73,
                       "AGB":  66}

    def __init__(self, id, time, source):
        self.id = id
        self.source = source
        self.values = {"time": time,
                       "id": id,
                       "complete": False}

    def is_complete(self):
        if len(self.values) == self.complete_values[self.source]:
            self.values["complete"] = True

    def set_value(self, parameter, value):
        self.values[parameter] = float(value)
        self.is_complete()

    def __getitem__(self, key):
        return self.values[key]


def parse_file(stdout_file, source):
    """

    :return:
    """
    complete_stars = []

    # keep track of which stars and timesteps are actively being parsed
    timesteps = defaultdict(dict)

    with open(stdout_file, "r") as stdout:
        for idx, line in enumerate(stdout):
            # if idx > 2000:
            #     continue
            if "detail_debug" not in line or source not in line:
                continue

            _, id, _, time_term, value_term = line.split(",")
            id = id.strip()
            time = time_term.split("=")[1].strip()
            parameter = value_term.split("=")[0].strip()
            value = value_term.split("=")[1].strip()

            if time in timesteps[id]:
                star = timesteps[id][time]
                star.set_value(parameter, value)

                # check if this completes the info for that star
                if star["complete"]:
                    complete_stars.append(star)

            else:  # adding a new timestep
                star = star_ejecta(id, float(time), source)
                timesteps[id][time] = star
                star.set_value(parameter, value)

        # we're now through the file, so return what we have
        return complete_stars
