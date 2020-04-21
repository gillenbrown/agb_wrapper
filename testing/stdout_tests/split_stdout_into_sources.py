agb_file = open("./stdout_agb.txt", "w")
snii_file = open("./stdout_snii.txt", "w")
snia_file = open("./stdout_snia.txt", "w")
wind_file = open("./stdout_wind.txt", "w")

in_file = open("./stdout_debug", "r")

for line in in_file:
    if "detail_debug" not in line:
        continue

    # figure out which source it belongs to. The comma is important, as
    # there are other references to the sources that is not what we want
    if "SNII," in line:
        snii_file.write(line)
    elif "SNIa," in line:
        snia_file.write(line)
    elif "AGB," in line:
        agb_file.write(line)
    elif "wind," in line:
        wind_file.write(line)

agb_file.close()
snii_file.close()
snia_file.close()
wind_file.close()
in_file.close()