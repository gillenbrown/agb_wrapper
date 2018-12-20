from collections import defaultdict



class star_ejecta(object):
    agb_values = 1
    snii_values = 1
    snia_values = 33
    wind_values = 1

    def __init__(self, id, time):
        self.id = id
        self.snia = {"time": time}
        self.snii = {"time": time}
        self.agb = {"time": time}
        self.winds = {"time": time}

    def is_complete(self):
        return (len(self.snia)  == self.snia_values and
                len(self.snii)  == self.snii_values and
                len(self.agb)   == self.agb_values  and
                len(self.winds) == self.wind_values)

    def set_value(self, source, parameter, value):
        if source == "SNII":
            parameter_dict = self.snii
        elif source == "SNIa":
            parameter_dict = self.snia
        elif source == "winds":
            parameter_dict = self.winds
        elif source == "AGB":
            parameter_dict = self.agb
        else:
            raise ValueError("Source {} not recognized.".format(source))

        parameter_dict[parameter] = float(value)


def parse_file(number):
    """

    :return:
    """

    complete_stars = []
    timesteps = defaultdict(dict)
    with open("../stdout", "r") as stdout:
        for idx, line in enumerate(stdout):
            # if idx > 2000:
            #     continue
            if "detail_debug" not in line:
                continue

            _, id, source, time_term, value_term = line.split(",")
            id = id.strip()
            source = source.strip()
            time = time_term.split("=")[1].strip()
            parameter = value_term.split("=")[0].strip()
            value = value_term.split("=")[1].strip()

            if time in timesteps[id]:
                star = timesteps[id][time]
                star.set_value(source, parameter, value)
                if star.is_complete():
                    complete_stars.append(star)
                    if len(complete_stars) == number:
                        return complete_stars

            else:
                star = star_ejecta(id, float(time))
                timesteps[id][time] = star
                star.set_value(source, parameter, value)

    # throw away sources that don't have all their information
    good_points = []
    for id in timesteps:
        for time in timesteps[id]:
            star = timesteps[id][time]
            if star.is_complete():
                good_points.append(star)

    return good_points

if __name__ == "__main__":
    parse_file()