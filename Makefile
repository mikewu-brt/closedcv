#
# Makefile to construct the Debian packaging layout
# for closedCV and populate it
#

MKDIR := mkdir
CP := cp
RM := rm
QUIET := @

INSTALL_DIR := install
INSTALL_PATH := $(INSTALL_DIR)/usr/local/bin/depthcal

DIRS_TO_COLLECT := \
	libs           \
	misc_scripts   \
	setup_examples \
	calibration_scripts

ALL_SOURCE_FILES := $(foreach dir,$(DIRS_TO_COLLECT), $(wildcard $(dir)/*.py))
ALL_TARGET_FILES := $(foreach file,$(ALL_SOURCE_FILES), $(INSTALL_PATH)/$(file))

all: package

package: $(INSTALL_PATH) $(ALL_TARGET_FILES) 

$(ALL_TARGET_FILES): $(ALL_SOURCE_FILES)
	$(QUIET)$(MKDIR) -p $(dir $@)
	$(QUIET)$(CP) $(subst $(INSTALL_PATH)/,,$@) $@

$(INSTALL_PATH):
	$(QUIET)$(MKDIR) -p $@
	$(QUIET)$(foreach dir,$(DIRS_TO_COLLECT), $(shell $(MKDIR) -p $(INSTALL_PATH)/$(dir)))

clean:
	$(QUIET)$(RM) -fr $(INSTALL_DIR)
	$(QUIET)$(RM) -fr *.deb

