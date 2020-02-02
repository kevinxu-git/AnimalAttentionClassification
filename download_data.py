from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1YbsRgf3MxtvyQIZ4yKb91FemaKPJv_E1',
                                    dest_path='./data/data.zip',
                                    unzip=True,
                                    showsize=True)
