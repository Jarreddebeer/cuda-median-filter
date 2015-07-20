#link
cuda: $(NVOBJS)
	$(NVCC) -g -G $(NVCLFLAGS) $^ -o $@ $(OMPLFLAGS) $(LFLAGS)
	
#Compile cu
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<

test: bin
	./$(BIN)

run: bin
	./$(BIN)

