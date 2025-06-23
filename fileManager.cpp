//
// Created by mihnea on 03.03.2025.
//

#include "fileManager.hpp"

#include <dirent.h>
#include <cstdio>  // For FILE, popen, pclose

// Function to open file dialog and get filename
bool openFileDlg(std::string& filename) {
    filename = "";
    const char* cmd = "zenity --file-selection --title=\"Select an image file\" 2>/dev/null";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return false;

    char buffer[1024];
    if (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        filename = buffer;
        if (!filename.empty() && filename[filename.length()-1] == '\n') {
            filename.erase(filename.length()-1);
        }
    }
    pclose(pipe);
    return !filename.empty();
}

// Function to open folder dialog and get folder name
bool openFolderDlg(std::string& folderName) {
    folderName = "";
    const char* cmd = "zenity --file-selection --directory --title=\"Select a folder\" 2>/dev/null";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return false;

    char buffer[1024];
    if (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        folderName = buffer;
        // Remove newline character if present
        if (!folderName.empty() && folderName[folderName.length()-1] == '\n') {
            folderName.erase(folderName.length()-1);
        }
    }
    pclose(pipe);
    return !folderName.empty();
}

// FileGetter class method implementations
FileGetter::FileGetter(const std::string& folderPath, const std::string& ext)
    : folderPath(folderPath), extension(ext) {
    dir = opendir(folderPath.c_str());
}

FileGetter::~FileGetter() {
    if (dir) closedir(static_cast<DIR*>(dir));
}

bool FileGetter::getNextAbsFile(std::string& filePath) {
    if (!dir) return false;

    struct dirent* entry;
    while ((entry = readdir(static_cast<DIR*>(dir))) != NULL) {
        std::string fname = entry->d_name;

        // Check if file has the specified extension
        size_t pos = fname.rfind(".");
        if (pos != std::string::npos) {
            std::string fileExt = fname.substr(pos + 1);
            if (fileExt == extension) {
                currentFile = fname;
                filePath = folderPath + "/" + fname;
                return true;
            }
        }
    }
    return false;
}

std::string FileGetter::getFoundFileName() {
    return currentFile;
}