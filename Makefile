CC = mpic++ -w
CXXFLAGS = --std=c++17 -O3 -I./src
LDFLAGS = 

# Targets
.PHONY: all clean

all: benchSddmm benchSpMM benchBoth serialSim numCheckSeq conv2bin


benchSddmm: miniapp/bench_sddmm.cpp src/*.cpp
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

benchSpMM: miniapp/bench_spmm.cpp src/*.cpp
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

benchBoth: miniapp/main_combined.cpp src/*.cpp
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

serialSim: miniapp/serialSim.cpp
	g++ -O3 -o $@ $^

numCheckSeq: miniapp/numCheckSeq.cpp src/core_ops.cpp
	g++ -O3 -o $@ $^ -I./src

conv2bin: tools/read_convert_mm.cpp
	g++ -o $@ $^ -I./src

clean:
	rm -f benchSddmm benchSpMM benchBoth serialSim numCheckSeq conv2bin 
