TARGETS:= xsens_read_ids
OBJLIBS	= xspublic
INCLUDE=-I. -Ixspublic
CFLAGS=-g $(INCLUDE)
CXXFLAGS=-std=c++11 $(CFLAGS)
LFLAGS=-Lxspublic/xscontroller -Lxspublic/xscommon -Lxspublic/xstypes -lxscontroller -lxscommon -lxstypes -lpthread -lrt -ldl

.PHONY: $(OBJLIBS)
all : $(OBJLIBS) $(TARGETS)

xspublic :
	$(MAKE) -C xspublic $(MFLAGS)

xsens_read_ids: xsens_read_ids.cpp.o

$(TARGETS):
	$(CXX) $(CFLAGS) $(INCLUDE) $^ -o $@ $(LFLAGS)

%.cpp.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean :
	-$(RM) $(OBJECTS) $(TARGETS) *.o
	-$(MAKE) -C xspublic/xscontroller $(MFLAGS) clean
	-$(MAKE) -C xspublic/xscommon $(MFLAGS) clean
	-$(MAKE) -C xspublic/xstypes $(MFLAGS) clean
