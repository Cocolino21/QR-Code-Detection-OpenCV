//
// Created by mihnea on 03.03.2025.
//

#ifndef FILEMANAGER_HPP
#define FILEMANAGER_HPP

#include <string>

// Function to open file dialog and get filename
bool openFileDlg(std::string& filename);

// Function to open folder dialog and get folder name
bool openFolderDlg(std::string& folderName);

// FileGetter class to iterate through files in a directory
class FileGetter {
private:
    void* dir;
    std::string folderPath;
    std::string extension;
    std::string currentFile;

public:
    FileGetter(const std::string& folderPath, const std::string& ext);
    ~FileGetter();

    bool getNextAbsFile(std::string& filePath);
    std::string getFoundFileName();
};

#endif // FILEMANAGER_HPP