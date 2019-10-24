from collections import defaultdict



class star_ejecta(object):
    snii_values = 54
    snia_values = 2
    wind_values = 2
    agb_values = 2

    def __init__(self, id, time):
        self.id = id
        self.snia = {"time": time}
        self.snii = {"time": time}
        self.wind = {"time": time}
        self.agb =  {"time": time}

        self.complete_wind = False
        self.complete_snii = False
        self.complete_snia = False
        self.complete_agb  = False

    def is_complete(self):
        if len(self.snia)  == self.snia_values:
            self.complete_snia = True
        if len(self.snii)  == self.snii_values:
            self.complete_snii = True
        if len(self.wind) == self.wind_values:
            self.complete_wind = True
        if len(self.agb) == self.agb_values:
            self.complete_agb = True

    def set_value(self, source, parameter, value):
        if source == "SNIa":
            parameter_dict = self.snia
        elif source == "SNII":
            parameter_dict = self.snii
        elif source == "Wind":
            parameter_dict = self.wind
        elif source == "AGB":
            parameter_dict = self.agb
        else:
            raise ValueError("Source {} not recognized.".format(source))

        parameter_dict[parameter] = float(value)

        self.is_complete()


def parse_file(stdout_file, max_number):
    """

    :return:
    """

    complete_stars_snii_inactive = []
    complete_stars_snii_active = []
    complete_stars_snia_inactive = []
    complete_stars_snia_active = []
    complete_stars_wind = []
    complete_stars_agb = []

    timesteps = defaultdict(dict)
    with open(stdout_file, "r") as stdout:
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

                # check if this completes the info for that star
                if star.complete_snia and \
                        len(complete_stars_snia_inactive) < max_number and \
                        star.snia["energy added"] == 0:
                    complete_stars_snia_inactive.append(star)
                if star.complete_snia and \
                        len(complete_stars_snia_active) < max_number and \
                        star.snia["energy added"] > 0:
                    complete_stars_snia_active.append(star)
                if star.complete_snii and \
                        len(complete_stars_snii_inactive) < max_number and \
                        star.snii["energy added"] == 0:
                    complete_stars_snii_inactive.append(star)
                if star.complete_snii and \
                        len(complete_stars_snii_active) < max_number and \
                        star.snii["energy added"] > 0:
                    complete_stars_snii_active.append(star)


                if star.complete_wind and len(complete_stars_wind) < max_number:
                    complete_stars_wind.append(star)
                if star.complete_agb  and len(complete_stars_agb)  < max_number:
                    complete_stars_agb.append(star)

                # check if we've obtained enough for the user
                if len(complete_stars_snia_inactive) == max_number and \
                   len(complete_stars_snia_active) == max_number and \
                   len(complete_stars_snii_inactive) == max_number and \
                   len(complete_stars_snii_active) == max_number and \
                   len(complete_stars_wind) == max_number and \
                   len(complete_stars_agb)  == max_number:

                    return complete_stars_wind, \
                           complete_stars_snii_inactive, \
                           complete_stars_snii_active, \
                           complete_stars_snia_inactive, \
                           complete_stars_snia_active, \
                           complete_stars_agb

            else:  # adding a new timestep
                star = star_ejecta(id, float(time))
                timesteps[id][time] = star
                star.set_value(source, parameter, value)

    # we might have returned earlier in the function if we had enough of all
    # sources. If we got here we didn't have enough, so we'll just return
    # what we have
    return complete_stars_wind, \
           complete_stars_snii_inactive, \
           complete_stars_snii_active, \
           complete_stars_snia_inactive, \
           complete_stars_snia_active, \
           complete_stars_agb

if __name__ == "__main__":
    parse_file(1E10)