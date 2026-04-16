# 加载必要的包
library(xml2)
library(openxlsx)
library(readr)

# 设置目标路径
folder_path <- "D:/NLPtools/ZCTC WordSmith edition"

# 获取所有 .txt 文件路径
txt_files <- list.files(path = folder_path, pattern = "\\.txt$", full.names = TRUE)

# 初始化结果数据框
genre_data <- data.frame(FileID = character(), Genre = character(), stringsAsFactors = FALSE)

# 遍历所有文件
for (file in txt_files) {
  file_name <- tools::file_path_sans_ext(basename(file))  # 提取不含扩展名的文件名
  
  # 读取UTF-16格式的XML文件内容
  xml_text <- read_file(file, locale = locale(encoding = "UTF-16LE"))
  
  # 解析XML
  tryCatch({
    doc <- read_xml(xml_text)
    genre <- xml_attr(xml_find_first(doc, "//cesDoc"), "genre")
    genre <- trimws(genre)  # 去除前后空格
  })
  
  # 添加到结果表
  genre_data <- rbind(genre_data, data.frame(FileID = file_name, Genre = genre, stringsAsFactors = FALSE))
}

# 写入Excel
output_path <- file.path(folder_path, "ZCTC_genre_list.xlsx")
write.xlsx(genre_data, file = output_path, rowNames = FALSE)

cat("✅ Excel 成功保存至：", output_path, "\n")

#####LCMC######
# 加载必要的包
library(xml2)
library(openxlsx)

# 设置目标路径
folder_path <- "D:/NLPtools/LCMCv2 WordSmith edition"

# 获取所有 .txt 文件路径
txt_files <- list.files(path = folder_path, pattern = "\\.txt$", full.names = TRUE)

# 初始化结果数据框
genre_data <- data.frame(FileID = character(), Genre = character(), stringsAsFactors = FALSE)

# 遍历所有文件
for (file in txt_files) {
  file_name <- tools::file_path_sans_ext(basename(file))  # 提取不含扩展名的文件名
  
  # 读取UTF-16格式的XML文件内容
  xml_text <- read_file(file, locale = locale(encoding = "UTF-16LE"))
  
  # 解析XML
  tryCatch({
    doc <- read_xml(xml_text)
    genre <- xml_attr(xml_find_first(doc, "//cesDoc"), "genre")
    genre <- trimws(genre)  # 去除前后空格
  })
  
  # 添加到结果表
  genre_data <- rbind(genre_data, data.frame(FileID = file_name, Genre = genre, stringsAsFactors = FALSE))
}

# 写入Excel
output_path <- file.path(folder_path, "LCMC_genre_list.xlsx")
write.xlsx(genre_data, file = output_path, rowNames = FALSE)

cat("✅ Excel 成功保存至：", output_path, "\n")

#####clean data##
###ZCTC
library(xml2)
library(stringr)

# 设置文件夹路径
folder_path <- "D:/NLPtools/ZCTC WordSmith edition"

# 获取所有 .txt 文件路径
txt_files <- list.files(path = folder_path, pattern = "\\.txt$", full.names = TRUE)

# 遍历每个文件
for (file in txt_files) {
  # 提取原文件名（不含扩展名）
  base_name <- tools::file_path_sans_ext(basename(file))
  
  # 设置输出文件名：ZCTC_A01_raw.txt
  output_file <- file.path(folder_path, paste0(base_name, "_raw.txt"))
  
  # 读取并解析 UTF-16 XML 内容
  xml_text <- read_file(file, locale = locale(encoding = "UTF-16LE"))
  doc <- read_xml(xml_text)
  
  # 提取所有 <s> 元素内容
  sentences <- xml_find_all(doc, ".//s")
  
  # 初始化句子列表
  clean_sentences <- c()
  
  for (s in sentences) {
    raw <- xml_text(s)  # 如：戴维·麦克林_nrf 时_ng ...
    cleaned <- str_replace_all(raw, "_[^\\s]+", "")  # 删除词性标签
    cleaned <- str_squish(cleaned)  # 删除多余空格
    clean_sentences <- c(clean_sentences, cleaned)
  }
  
  # 保存到对应的 _raw.txt 文件
  writeLines(clean_sentences, output_file, useBytes = TRUE)
  cat("✅ 已生成：", output_file, "\n")
}

###LCMC
library(xml2)
library(stringr)

# 设置文件夹路径
folder_path <- "D:/NLPtools/LCMCv2 WordSmith edition"

# 获取所有 .txt 文件路径
txt_files <- list.files(path = folder_path, pattern = "\\.txt$", full.names = TRUE)

# 遍历每个文件
for (file in txt_files) {
  # 提取原文件名（不含扩展名）
  base_name <- tools::file_path_sans_ext(basename(file))
  
  # 设置输出文件名：ZCTC_A01_raw.txt
  output_file <- file.path(folder_path, paste0(base_name, "_raw.txt"))
  
  # 读取并解析 UTF-16 XML 内容
  xml_text <- read_file(file, locale = locale(encoding = "UTF-16LE"))
  doc <- read_xml(xml_text)
  
  # 提取所有 <s> 元素内容
  sentences <- xml_find_all(doc, ".//s")
  
  # 初始化句子列表
  clean_sentences <- c()
  
  for (s in sentences) {
    raw <- xml_text(s)  # 如：戴维·麦克林_nrf 时_ng ...
    cleaned <- str_replace_all(raw, "_[^\\s]+", "")  # 删除词性标签
    cleaned <- str_squish(cleaned)  # 删除多余空格
    clean_sentences <- c(clean_sentences, cleaned)
  }
  
  # 保存到对应的 _raw.txt 文件
  writeLines(clean_sentences, output_file, useBytes = TRUE)
  cat("✅ 已生成：", output_file, "\n")
}

####ZCTC entropy###
library(quanteda)
library(quanteda.textstats)
library(openxlsx)
library(stringr)

# 设置文件夹路径
folder_path <- "D:/NLPtools/ZCTC WordSmith edition"

# 获取所有 *_raw.txt 文件路径
raw_files <- list.files(path = folder_path, pattern = "_raw\\.txt$", full.names = TRUE)

# 初始化结果数据框
entropy_results <- data.frame(FileID = character(), WRDentropy = numeric(), stringsAsFactors = FALSE)

# 遍历每个文件
for (file in raw_files) {
  # 提取文件名（如 ZCTC_A01）
  file_id <- str_replace(basename(file), "_raw\\.txt$", "")
  
  # 读取文本内容
  text <- readLines(file, warn = FALSE, encoding = "UTF-8")
  text <- paste(text, collapse = " ")  # 合并为一段文本
  
  # 创建语料库
  corpus_obj <- corpus(text)
  toks <- tokens(corpus_obj, remove_punct = TRUE)  # 去除标点
  dfmat <- dfm(toks)
  
  # 计算熵
  entropy_val <- textstat_entropy(dfmat)$entropy
  
  # 添加到结果
  entropy_results <- rbind(entropy_results, data.frame(FileID = file_id, WRDentropy = entropy_val))
}

# 保存为Excel
output_path <- file.path(folder_path, "ZCTC_WRDentropy_results.xlsx")
write.xlsx(entropy_results, file = output_path, rowNames = FALSE)

cat("✅ WRDentropy计算完成，结果保存至：", output_path, "\n")


####LCMC entropy###
library(quanteda)
library(quanteda.textstats)
library(openxlsx)
library(stringr)

# 设置文件夹路径
folder_path <- "D:/NLPtools/LCMCv2 WordSmith edition"

# 获取所有 *_raw.txt 文件路径
raw_files <- list.files(path = folder_path, pattern = "_raw\\.txt$", full.names = TRUE)

# 初始化结果数据框
entropy_results <- data.frame(FileID = character(), WRDentropy = numeric(), stringsAsFactors = FALSE)

# 遍历每个文件
for (file in raw_files) {
  # 提取文件名（如 ZCTC_A01）
  file_id <- str_replace(basename(file), "_raw\\.txt$", "")
  
  # 读取文本内容
  text <- readLines(file, warn = FALSE, encoding = "UTF-8")
  text <- paste(text, collapse = " ")  # 合并为一段文本
  
  # 创建语料库
  corpus_obj <- corpus(text)
  toks <- tokens(corpus_obj, remove_punct = TRUE)  # 去除标点
  dfmat <- dfm(toks)
  
  # 计算熵
  entropy_val <- textstat_entropy(dfmat)$entropy
  
  # 添加到结果
  entropy_results <- rbind(entropy_results, data.frame(FileID = file_id, WRDentropy = entropy_val))
}

# 保存为Excel
output_path <- file.path(folder_path, "LCMC_WRDentropy_results.xlsx")
write.xlsx(entropy_results, file = output_path, rowNames = FALSE)

cat("✅ WRDentropy计算完成，结果保存至：", output_path, "\n")

##merge table
library(readxl)
library(dplyr)
library(openxlsx)

# 设置路径
folder_path <- "D:/NLPtools"

# 读取四个表格
zctc_genre <- read_excel(file.path(folder_path, "ZCTC_genre_list.xlsx"))
zctc_entropy <- read_excel(file.path(folder_path, "ZCTC_WRDentropy_results.xlsx"))

lcmc_genre <- read_excel(file.path(folder_path, "LCMC_genre_list.xlsx"))
lcmc_entropy <- read_excel(file.path(folder_path, "LCMC_WRDentropy_results.xlsx"))

# 合并 ZCTC 表
zctc_merged <- merge(zctc_genre, zctc_entropy, by = "FileID", all = TRUE)

# 合并 LCMC 表
lcmc_merged <- merge(lcmc_genre, lcmc_entropy, by = "FileID", all = TRUE)

# 合并两个数据框
final_table <- bind_rows(zctc_merged, lcmc_merged)

# 导出为 Excel 文件
output_path <- file.path(folder_path, "Combined_Genre_WRDentropy.xlsx")
write.xlsx(final_table, file = output_path, rowNames = FALSE)

cat("✅ 合并完成，结果保存至：", output_path, "\n")

##generate new data
library(readxl)
library(dplyr)
library(openxlsx)
library(stringr)

# 读取原始数据
data <- read_excel("D:/NLPtools/Combined_Genre_WRDentropy.xlsx")

# 添加 Corpus 列
data <- data %>%
  mutate(Corpus = case_when(
    str_detect(FileID, "ZCTC") ~ "ZCTC",
    str_detect(FileID, "LCMC") ~ "LCMC",
    TRUE ~ "Unknown"
  ))

# 标准化 Genre 列
data <- data %>%
  mutate(Genre = str_trim(Genre),
         Genre = case_when(
           Genre == "Academic prose" ~ "Academic prose",
           Genre %in% c("News reportage", "News editorial", "News review") ~ "Press",
           Genre %in% c("Religious", "Skill/trade/hobby", "Popular lore", 
                        "Biography and essay", "Miscellaneous (official document, report ect)") ~ "General Prose",
           TRUE ~ "Fiction"
         ))

# 保存结果
output_path <- "D:/NLPtools/Combined_Genre_WRDentropy_cleaned.xlsx"
write.xlsx(data, file = output_path, rowNames = FALSE)

cat("✅ 数据整理完成，已保存至：", output_path, "\n")

##two-way anova
WRDentropy <- read_excel("D:/NLPtools/Combined_Genre_WRDentropy_cleaned.xlsx")

anova2<-aov(WRDentropy~
              as.factor(Genre)*as.factor(Corpus),data=WRDentropy)

res<-anova2$residuals
hist(res,main="Histogram of
residuals",xlab="Residuals")

shapiro.test(res)

par(mfrow=c(1, 2)) # make the plotting window have 1 rows & 2 columns
(bcn.parameters <- car::powerTransform(WRDentropy$WRDentropy ~ 1, family="bcnPower"))

library(car)
hist(WRDentropy$WRDentropy.bcn.sc <- # plot a histogram of this new variable,
       as.numeric(scale(bcnPower( # the scaled version of the power-transformed
         WRDentropy$WRDentropy, # FREQ values, which are transformed w/
         lambda=bcn.parameters[[1]], # this lambda &
         gamma=bcn.parameters[[2]]))), # this gamma
     main="", xlab="Power transform") # no heading, but an x-axis label
plot(jitter(WRDentropy$WRDentropy), jitter(WRDentropy$WRDentropy.bcn.sc), pch=16, col="#00000030"); grid()


anova2<-aov(WRDentropy.bcn.sc~
              as.factor(Genre)*as.factor(Corpus),data=WRDentropy)

res<-anova2$residuals
hist(res,main="Histogram of
residuals",xlab="Residuals")

shapiro.test(res)

leveneTest(WRDentropy.bcn.sc~as.factor(Genre)*as.factor(Corpus),data=WRDentropy)


WRDentropy$Genre <- factor(WRDentropy$Genre, levels = c("Press", "General Prose", "Academic prose", "Fiction"))
WRDentropy$Corpus <- factor(WRDentropy$Corpus, levels = c("LCMC", "ZCTC"))

model<-lm(WRDentropy~
              Genre*Corpus,data=WRDentropy,
          contrasts = list(Genre=contr.helmert,
                           Corpus=contr.helmert))

Anova(model, type=3,white.adjust = TRUE)

res<-model$residuals
hist(res,main="Histogram of
residuals",xlab="Residuals")

shapiro.test(res)

library(emmeans)
emmeans(model, pairwise ~ Corpus | Genre)

library(ggplot2)

# 保证变量是因子
WRDentropy$Genre <- factor(WRDentropy$Genre, levels = c("Press", "General Prose", "Academic prose", "Fiction"))
WRDentropy$Corpus <- factor(WRDentropy$Corpus, levels = c("LCMC", "ZCTC"))

# 绘图
ggplot(WRDentropy, aes(Genre, WRDentropy.bcn.sc, fill = Corpus)) +
  geom_boxplot(position = position_dodge(width = 0.75), width = 0.6, outlier.shape = NA) +  # boxplot 不显示离群值
  geom_jitter(aes(fill = Corpus), 
              position = position_jitterdodge(jitter.width = 0.25, dodge.width = 0.75), 
              alpha = 0.7, size = 1.5, shape = 21, stroke = 0.2) +
  scale_color_manual(values = c("#00BFC4", "#FBAF44")) +
  scale_fill_manual(values = c("#00BFC4", "#FBAF44")) +
  labs(y = "Unigram Entropy", x = "Genre") +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(angle = 15, hjust = 1)
  )


