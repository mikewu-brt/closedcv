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
CAL_INSTALL_FILE_PATH := $(INSTALL_DIR)/usr/share/light-depth/

DIRS_TO_COLLECT := \
	libs           \
	misc_scripts   \
	setup_examples \
	calibration_scripts

JSON_CALIBRATION_DIRS_TO_COLLECT := cal-files

ALL_SOURCE_FILES := $(foreach dir,$(DIRS_TO_COLLECT), $(wildcard $(dir)/*.py))
ALL_TARGET_FILES := $(foreach file,$(ALL_SOURCE_FILES), $(INSTALL_PATH)/$(file))
ALL_SOURCE_CALIBRATION_FILES := $(foreach dir,$(JSON_CALIBRATION_DIRS_TO_COLLECT), $(wildcard $(dir)/*.json))
ALL_TARGET_CALIBRATION_FILES := $(foreach file,$(ALL_SOURCE_CALIBRATION_FILES), $(CAL_INSTALL_FILE_PATH)/$(file))

all: package

package: $(INSTALL_PATH) $(CAL_INSTALL_FILE_PATH) $(ALL_TARGET_FILES) $(ALL_TARGET_CALIBRATION_FILES)

$(ALL_TARGET_FILES): $(ALL_SOURCE_FILES)
	$(QUIET)$(MKDIR) -p $(dir $@)
	$(QUIET)$(CP) $(subst $(INSTALL_PATH)/,,$@) $@

$(INSTALL_PATH):
	$(QUIET)$(MKDIR) -p $@
	$(QUIET)$(foreach dir,$(DIRS_TO_COLLECT), $(shell $(MKDIR) -p $(INSTALL_PATH)/$(dir)))

$(ALL_TARGET_CALIBRATION_FILES): $(ALL_SOURCE_CALIBRATION_FILES)
	$(QUIET)$(MKDIR) -p $(dir $@)
	$(QUIET)$(CP) $(subst $(CAL_INSTALL_FILE_PATH)/,,$@) $@

$(CAL_INSTALL_FILE_PATH):
	$(QUIET)$(MKDIR) -p $@

clean:
	$(QUIET)$(RM) -fr $(INSTALL_DIR)
	$(QUIET)$(RM) -fr *.deb

