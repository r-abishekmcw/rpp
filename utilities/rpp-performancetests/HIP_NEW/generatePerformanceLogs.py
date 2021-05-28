import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--profiling', type=str, default='NO', help='Run with profiler? - (YES/NO)')
args = parser.parse_args()

profilingOption = args.profiling

if profilingOption == "NO":

    subprocess.call("./rawLogsGenScript.sh", shell=True)

    log_file_list = [
        "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW/BatchPD_hip_pkd3_hip_raw_performance_log.txt"
        "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW/BatchPD_hip_pln3_hip_raw_performance_log.txt",
        "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW/BatchPD_hip_pln1_hip_raw_performance_log.txt"
        ]

    for log_file in log_file_list:

        # Opening log file
        f = open(log_file,"r")
        print("\n\n\nOpened log file -> ", log_file)

        stats = []
        maxVals = []
        minVals = []
        avgVals = []
        functions = []
        prevLine = ""

        # Loop over each line
        for line in f:
            if "max,min,avg" in line:
                split_word_start = "Running "
                split_word_end = " 100"
                prevLine = prevLine.partition(split_word_start)[2].partition(split_word_end)[0]
                if prevLine not in functions:
                    functions.append(prevLine)
                    split_word_start = "max,min,avg = "
                    split_word_end = "\n"
                    stats = line.partition(split_word_start)[2].partition(split_word_end)[0].split(",")
                    maxVals.append(stats[0])
                    minVals.append(stats[1])
                    avgVals.append(stats[2])

            if line != "\n":
                prevLine = line

        # Print log lengths
        print("Functionalities - ", len(functions))

        # Print summary of log
        print("\n\nFunctionality\t\t\t\t\t\t\t\tFrames Count\tmax(s)\t\tmin(s)\t\tavg(s)\n")
        maxCharLength = len(max(functions, key=len))
        functions = [x + (' ' * (maxCharLength - len(x))) for x in functions]
        for i, func in enumerate(functions):
            print(func, "\t100\t\t", maxVals[i], "\t", minVals[i], "\t", avgVals[i])

        # Closing log file
        f.close()

elif profilingOption == "YES":

    subprocess.call("./rawLogsGenScript_withProfiling.sh", shell=True)

    RESULTS_DIR = "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW"
    print("RESULTS_DIR = " + RESULTS_DIR)
    CONSOLIDATED_FILE_PKD3 = RESULTS_DIR + "/consolidated_results_pkd3.stats.csv"
    CONSOLIDATED_FILE_PLN1 = RESULTS_DIR + "/consolidated_results_pln1.stats.csv"
    CONSOLIDATED_FILE_PLN3 = RESULTS_DIR + "/consolidated_results_pln3.stats.csv"

    TYPE_LIST = ["PKD3", "PLN1", "PLN3"]
    CASE_NUM_LIST = range(0, 82, 1)
    BIT_DEPTH_LIST = range(0, 7, 1)
    OFT_LIST = range(0, 2, 1)
    d_counter = {"PKD3":0, "PLN1":0, "PLN3":0}

    for TYPE in TYPE_LIST:

        new_file = open(RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv",'w')
        new_file.write('"HIP Kernel Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')

        for CASE_NUM in CASE_NUM_LIST:
            CASE_RESULTS_DIR = RESULTS_DIR + "/" + TYPE + "/case_" + str(CASE_NUM)
            print("CASE_RESULTS_DIR = " + CASE_RESULTS_DIR)

            for BIT_DEPTH in BIT_DEPTH_LIST:

                for OFT in OFT_LIST:

                    CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + "_bitDepth" + str(BIT_DEPTH) + "_oft" + str(OFT) + ".stats.csv"
                    print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                    try:
                        case_file = open(CASE_FILE_PATH,'r')
                        for line in case_file:
                            print(line)
                            if not(line.startswith('"Name"')):
                                new_file.write(line)
                                d_counter[TYPE] = d_counter[TYPE] + 1
                        case_file.close()
                    except IOError:
                        print("Unable to open case results")
                        continue

        new_file.close()
        os.system('chown $USER:$USER ' + RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")

    try:
        import pandas as pd
        pd.options.display.max_rows = None

        for TYPE in TYPE_LIST:
            print("\n\n\nKernels tested - ", d_counter[TYPE], "\n\n")
            df = pd.read_csv(RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")
            df["AverageMs"] = df["AverageNs"] / 1000000
            dfPrint = df.drop(['Percentage'], axis=1)
            dfPrint["HIP Kernel Name"] = dfPrint.iloc[:,0].str.lstrip("Hip_")
            print(dfPrint)

    except ImportError:
        print("\nPandas not available! Results of GPU profiling experiment are available in " + CONSOLIDATED_FILE)

    except IOError:
        print("Unable to open results in " + RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")


