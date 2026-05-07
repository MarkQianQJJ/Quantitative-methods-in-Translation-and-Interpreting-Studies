from PgsFile import maketmx

maketmx("zh_en.tmx",["你好","再见"],["Hello","Goodbye"],"zh-CN","en-US", client_name="SISU", project_id="202500111Z", domain="Life")
print("Complete!")