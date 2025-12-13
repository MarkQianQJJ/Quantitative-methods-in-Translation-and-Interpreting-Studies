import pyter

#https://pypi.org/project/pyter/

# 参考翻译和机器翻译输出
reference = "This is a test sentence."
system_output = "This is the test sentence."

# 计算 TER
ter_score = pyter.ter(reference, system_output)
print(f"TER score: {ter_score}")
