CC=mpic++ -w

testSDDMM: 
	g++ -ggdb -o testSDDMM miniapp/main_random_mat.cpp src/*.cpp -I/usr/include/x86_64-linux-gnu/mpich -L/usr/lib/x86_64-linux-gnu -lmpi
sddmmbin: 
	$(CC) --std=c++17 -O3 -o sddmmbin miniapp/main_binary_mm.cpp src/*.cpp -I./src
benchSddmm: 
	$(CC) --std=c++17 -O3 -o bench_sddmm miniapp/bench_sddmm.cpp src/*.cpp -I./src
benchSpMM: 
	$(CC) --std=c++17 -O3 -o bench_spmm miniapp/bench_spmm.cpp src/*.cpp -I./src
	#g++ -O3 -o sddmm miniapp/main.cpp src/*.cpp -I/usr/include/x86_64-linux-gnu/mpich -L/usr/lib/x86_64-linux-gnu -lmpi
spmm:
	$(CC) --std=c++17 -O3 -o spmm miniapp/main_spmm.cpp src/*.cpp -I./src
benchBoth:
	$(CC) --std=c++17 -O3 -o benchBoth miniapp/main_combined.cpp src/*.cpp -I./src
#	$(CC) --std=c++17 -g -O3 -o spmm miniapp/main_spmm.cpp src/*.cpp -I./src -L/home/nabil/.local/lib -lmpiP
countExcess:
	$(CC) --std=c++17 -O3 -o countExcess miniapp/countExcess.cpp src/*.cpp -I./src
debugSDDMM: 
	$(CC) --std=c++17 -ggdb -o sddmmbin miniapp/main.cpp src/*.cpp -I./src
debugSPMM: 
	$(CC) --std=c++17 -ggdb -o spmm miniapp/main_spmm.cpp src/*.cpp -I./src
	#g++ -O3 -o sddmm miniapp/main.cpp src/*.cpp -I/usr/include/x86_64-linux-gnu/mpich -L/usr/lib/x86_64-linux-gnu -lmpi
testBin: 
	$(CC) --std=c++17 -ggdb -o testbin miniapp/test_binary_mm.cpp src/*.cpp -I./src
serialSim: 
	g++ -O3 -o serialSim miniapp/serialSim.cpp

numCheckSeq: 
	g++ -O3 -o numCheckSeq miniapp/numCheckSeq.cpp src/core_ops.cpp -I./src

conv2bin:
	g++ -ggdb -o conv2bin tools/read_convert_mm.cpp -I./src
clean: 
	rm sddmmbin sddmm serialSim debug
