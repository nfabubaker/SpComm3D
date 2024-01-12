testSDDMM: 
	g++ -ggdb -o testSDDMM miniapp/main_random_mat.cpp src/*.cpp -I/usr/include/x86_64-linux-gnu/mpich -L/usr/lib/x86_64-linux-gnu -lmpi
sddmmbin: 
	mpic++ --std=c++17 -O3 -o sddmmbin miniapp/main_binary_mm.cpp src/*.cpp -I./src
sddmm: 
	mpic++ --std=c++17 -O3 -o sddmm miniapp/main.cpp src/*.cpp -I./src
	#g++ -O3 -o sddmm miniapp/main.cpp src/*.cpp -I/usr/include/x86_64-linux-gnu/mpich -L/usr/lib/x86_64-linux-gnu -lmpi
spmm:
	mpic++ --std=c++17 -O3 -o spmm miniapp/main_spmm.cpp src/*.cpp -I./src
#	mpic++ --std=c++17 -g -O3 -o spmm miniapp/main_spmm.cpp src/*.cpp -I./src -L/home/nabil/.local/lib -lmpiP
debugSDDMM: 
	mpic++ --std=c++17 -ggdb -o sddmmbin miniapp/main.cpp src/*.cpp -I./src
debugSPMM: 
	mpic++ --std=c++17 -ggdb -o spmm miniapp/main_spmm.cpp src/*.cpp -I./src
	#g++ -O3 -o sddmm miniapp/main.cpp src/*.cpp -I/usr/include/x86_64-linux-gnu/mpich -L/usr/lib/x86_64-linux-gnu -lmpi
testBin: 
	mpic++ --std=c++17 -ggdb -o testbin miniapp/test_binary_mm.cpp src/*.cpp -I./src
serialSim: 
	g++ -ggdb -o serialSim miniapp/serialSim.cpp

conv2bin:
	g++ -O3 -o conv2bin tools/read_convert_mm.cpp 
clean: 
	rm sddmmbin sddmm serialSim debug
