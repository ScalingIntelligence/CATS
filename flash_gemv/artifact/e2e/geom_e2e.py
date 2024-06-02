from scipy.stats.mstats import gmean
import argparse

def geom_e2e(input):
    with open(input, 'r') as f:
        data = f.readlines()
    # Each line is composed of 166,256,8573.2177734375,29.860433592759975
    # Geomean over the last column
    data = [float(x.split(',')[-1]) for x in data]
    return gmean(data)

def list_geom_e2e(input):
    outlist = []
    for i in [256,512,1024,2048]:
        with open(input+"-"+str(i)+".csv", 'r') as f:
            data = f.readlines()
        data = [float(x.split(',')[-1]) for x in data]
        outlist.append(gmean(data))
    print(*outlist, sep=",")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='E2e geomean.')
    parser.add_argument('--input', type=str, help='input file')
    args = parser.parse_args()
    # print(f"{args.input},{geom_e2e(args.input)}")
    list_geom_e2e(args.input)