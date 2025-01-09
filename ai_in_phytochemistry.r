##########################################
##### OCR CODE #######
##########################################
import cv2
import pytesseract
import pandas as pd
import numpy as np

image_path = "/project_data/shared/ai_in_phytochemistry/vingette_3_visionAI/articles/phytosterols/png_annot/10.1007-s11746-009-1384-5.png"
# image_path = "/project_data/shared/ai_in_phytochemistry/vingette_3_visionAI/articles/phytosterols/png_annot/10.1007-s11947-019-02341-8.png"
# image_path = "/project_data/shared/ai_in_phytochemistry/vingette_3_visionAI/articles/phytosterols/png_annot/10.1007-s12161-017-1111-z.png"

image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(thresh, config=custom_config)

lines = text.strip().split("\n")
data = [line.split() for line in lines]

df_ocr = pd.DataFrame(data)
# df_ocr

# df_llm = pd.read_csv("/project_data/shared/ai_in_phytochemistry/vingette_3_visionAI/articles/phytosterols/png_annot/10.1007-s11746-009-1384-5_extracted.csv")
# df_llm = pd.read_csv("/project_data/shared/ai_in_phytochemistry/vingette_3_visionAI/articles/phytosterols/png_annot/10.1007-s11947-019-02341-8_extracted.csv")
# df_llm = pd.read_csv("/project_data/shared/ai_in_phytochemistry/vingette_3_visionAI/articles/phytosterols/png_annot/10.1007-s12161-017-1111-z_extracted.csv")
# df_llm

##########################################
##### CO-OCCURRENCE CODE #######
##########################################

text <- read_csv("/project_data/shared/ai_in_phytochemistry/vingette_2_RAG/rag_output_processed_rep1.csv", show_col_types = FALSE)
text$compound <- gsub(" Found.*$", "", gsub("Is ", "", text$association))
text$species <- gsub(".* Found in ", "", text$association)
text$at <- paste0(text$title, ". ", text$abstract)
text$co_occurr <- NA
# colnames(text)
for ( i in 1:dim(text)[1] ) { # i=1
    text$co_occurr[i] <- all(c(
        grepl(tolower(text$compound[i]), tolower(text$at[i])),
        grepl(gsub("_", " ", tolower(text$species[i])), tolower(text$at[i]))
    ))
}
# select(text, compound, species, at, manual_evaluation, co_occurr, `gpt_response_you_are_an_ai_classifier_say_true_or_false~abstract`)[1,]
text$`gpt_response_you_are_an_ai_classifier_say_true_or_false~abstract` <- substr(as.character(text$`gpt_response_you_are_an_ai_classifier_say_true_or_false~abstract`), 1, 1)
text$co_occurr <- substr(as.character(text$co_occurr), 1, 1)

text %>% filter(manual_evaluation %in% c("T", "F")) %>%
    mutate(
        co_occurr_judgement = case_when(
            co_occurr == "T" & manual_evaluation == "F" ~ "false_positive",
            co_occurr == "F" & manual_evaluation == "T" ~ "false_negative",
            co_occurr == manual_evaluation & co_occurr == "T" ~ "true_positive",
            co_occurr == manual_evaluation & co_occurr == "F" ~ "true_negative",
            TRUE ~ "unknown"
        ),
        gpt_judgement = case_when(
            `gpt_response_you_are_an_ai_classifier_say_true_or_false~abstract` == "T" & manual_evaluation == "F" ~ "false_positive",
            `gpt_response_you_are_an_ai_classifier_say_true_or_false~abstract` == "F" & manual_evaluation == "T" ~ "false_negative",
            `gpt_response_you_are_an_ai_classifier_say_true_or_false~abstract` == manual_evaluation & `gpt_response_you_are_an_ai_classifier_say_true_or_false~abstract` == "T" ~ "true_positive",
            `gpt_response_you_are_an_ai_classifier_say_true_or_false~abstract` == manual_evaluation & `gpt_response_you_are_an_ai_classifier_say_true_or_false~abstract` == "F" ~ "true_negative",
            TRUE ~ "unknown"
        ),
    ) -> results

# head(select(results, compound, species, at, manual_evaluation, co_occurr, `gpt_response_you_are_an_ai_classifier_say_true_or_false~abstract`, co_occurr_judgement, gpt_judgement))

results %>%
    select(gpt_judgement, co_occurr_judgement) %>%
    pivot_longer(names_to = "approach", values_to = "outcome", cols = c(1:2)) %>%
    group_by(approach, outcome) %>% summarize(count = n()) %>%
    group_by(approach) %>% mutate(percent = round(count/sum(count)*100))

aspect = 1.5
pdf(
    file = "/project_data/shared/ai_in_phytochemistry/figures/figS1.pdf",
    width = 4 * aspect, height = 1.6 * aspect
)
    results %>%
        select(gpt_judgement, co_occurr_judgement) %>%
        pivot_longer(names_to = "approach", values_to = "outcome", cols = c(1:2)) %>%
        group_by(approach, outcome) %>% summarize(count = n()) %>%
        ggplot() +
            geom_col(
                aes(x = count, y = approach, fill = outcome),
                color = "black"
            ) +
            facet_grid(outcome~.) +
            scale_fill_manual(
                values = c("#9c2700", "#e00025", "#2dd413", "#1B9E77"),
                name = ""
            ) +
            scale_y_discrete(name = "", labels = c("Co-occurrence classification", "LLM classification")) +
            scale_x_continuous(name = "Number of articles") +
            theme_bw() +
            theme(
                strip.text.y = element_text(angle = 0)
            )
invisible(dev.off())
print(magick::image_read_pdf("/project_data/shared/ai_in_phytochemistry/figures/figS1.pdf"), info=F)

##########################################
##### MAIN MANUSCRIPT #######
##########################################

bustalab <- TRUE
source("https://thebustalab.github.io/phylochemistry/phylochemistry.R")
library(readxl)
library(dplyr)
library(ggupset)
library(cluster)

## Image of process
    process_image <- ggplot(data = data.frame(x = c(0,1), y = c(0.5,0.5))) +
    geom_point(aes(x = x, y = y), color = "white") +
    theme_void() +
    annotation_custom(
        rasterGrob(
            readPNG(
                "/project_data/shared/ai_in_phytochemistry/figures/FINAL Figure.png"
            ), interpolate=TRUE
        ), xmin=0, xmax=1, ymin=0, ymax=1
    )

## Results
    correct_answers <- read_csv("/project_data/shared/ai_in_phytochemistry/enzyme_searching_correct_answers.csv")
    # ggplot(cbind(data.frame(x = seq(1:142)), correct_answers)) + geom_tile(aes(x = x, y = Manual))
    # table(correct_answers[1:60,]$Manual); 49/11
    file_list <- list.files(path = "/project_data/shared/ai_in_phytochemistry/_pub_results/", pattern = "*.xlsx", full.names = TRUE)
    
    df_list <- list()
    for (i in 1:length(file_list)) {
        df <- read_excel(file_list[i])
        if("Response" %in% colnames(df)) { colnames(df)[colnames(df)=="Response"] <- "GPT_Response" }
        df$Model <- basename(file_list[i])
        df_list[[i]] <- df
    }

    merged_df <- do.call(rbind, df_list)
    merged_df$Response <- substr(merged_df$GPT_Response, 1, 1)
    # merged_df[grep("CoT1", merged_df$Model),]
    merged_df <- select(merged_df, Protein_ID, Model, Response)
    
    merged_df$Prompt <- gsub("~.*$", "", gsub("\\.xlsx", "", gsub(".*_", "", merged_df$Model)))
    merged_df$Replicate <- gsub(".*~", "", gsub("\\.xlsx", "", gsub(".*_", "", merged_df$Model)))
    merged_df$Model <- gsub("_.*$", "", merged_df$Model) # dim(merged_df)
    merged_df <- left_join(merged_df, correct_answers)
    merged_df$Manual <- substr(merged_df$Manual, 0, 1)
    
    merged_df <- filter(merged_df, Response %in% c("Y", "N"))
    merged_df$Correct <- merged_df$Manual == merged_df$Response
    merged_df <- mutate(merged_df, Category = case_when(
        Response == "Y" & Correct == TRUE ~ "True Positive",
        Response == "N" & Correct == TRUE ~ "True Negative",
        Response == "Y" & Correct == FALSE ~ "False Positive",
        Response == "N" & Correct == FALSE ~ "False Negative"
    ))

    # unique(merged_df$Model)
    merged_df$Model[merged_df$Model == "GPT3.5Turbo"] <- "gpt-3.5-turbo"
    merged_df$Model[merged_df$Model == "GPT40125Preview"] <- "gpt-4-0125-preview"
    merged_df$Model[merged_df$Model == "GPT42024"] <- "gpt-4o-2024-08-06"
    merged_df$Model[merged_df$Model == "GPT4o20240806"] <- "gpt-4o-2024-08-06"
    merged_df$Model[merged_df$Model == "GPT4omini"] <- "gpt-4o-mini"
    merged_df$Model[merged_df$Model == "GPT4Preview"] <- "gpt-4-0125-preview"

    # unique(merged_df$Prompt)
    merged_df$Prompt <- factor(merged_df$Prompt, levels = c("InitialSystemPrompt", "CoT1", "CoT3"))
    merged_df -> merged_df_raw
    # merged_df_raw %>% group_by(Model, Prompt, Replicate) %>% summarize(tot = n())
    merged_df %>%
        mutate(test_group = factor(paste0(Model, Prompt, Category))) %>%        
        group_by(Model, Prompt, Category, Replicate, test_group) %>%    
        summarize(Count = n()) -> merged_df
    merged_df -> merged_df_raw2
    # merged_df_raw %>% group_by(Model, Prompt, Replicate) %>% summarize(tot = sum(Count))

    merged_df$test_group <- gsub("-", "", merged_df$test_group)

    rbind(
        merged_df %>% ungroup() %>% filter(Category == "True Positive") %>%
        tukey_hsd(Count ~ test_group) %>% pGroups(),
        merged_df %>% ungroup() %>% filter(Category == "True Negative") %>%
        tukey_hsd(Count ~ test_group) %>% pGroups(),
        merged_df %>% ungroup() %>% filter(Category == "False Positive") %>%
        tukey_hsd(Count ~ test_group) %>% pGroups(),
        merged_df %>% ungroup() %>% filter(Category == "False Negative") %>%
        tukey_hsd(Count ~ test_group) %>% pGroups()
    ) -> merged_df_stats

    
    ######################################################################## random subsets
    sample_fraction <- 0.9
    set.seed(abs(round(rnorm(1)*100,0)))
    merged_df_stats_subset_1 <- rbind(
        merged_df[sample((1:dim(merged_df)[1]), (sample_fraction*dim(merged_df)[1])),] %>%
            ungroup() %>% filter(Category == "True Positive") %>% tukey_hsd(Count ~ test_group) %>% 
            mutate(
                p.adj.signif_1 = case_when(p.adj.signif == "ns" ~ "ns", p.adj.signif != "ns" ~ "sig"),
                pval_1 = p.adj
            ),
        merged_df %>% 
            ungroup() %>% filter(Category == "True Negative") %>% tukey_hsd(Count ~ test_group) %>% 
            mutate(
                p.adj.signif_1 = case_when(p.adj.signif == "ns" ~ "ns", p.adj.signif != "ns" ~ "sig"),
                pval_1 = p.adj
            ),
        merged_df %>% 
            ungroup() %>% filter(Category == "False Positive") %>% tukey_hsd(Count ~ test_group) %>% 
            mutate(
                p.adj.signif_1 = case_when(p.adj.signif == "ns" ~ "ns", p.adj.signif != "ns" ~ "sig"),
                pval_1 = p.adj
            ),
        merged_df %>% 
            ungroup() %>% filter(Category == "False Negative") %>% tukey_hsd(Count ~ test_group) %>% 
            mutate(
                p.adj.signif_1 = case_when(p.adj.signif == "ns" ~ "ns", p.adj.signif != "ns" ~ "sig"),
                pval_1 = p.adj
            )
    )
    set.seed(abs(round(rnorm(1)*100,0)))
    merged_df_stats_subset_2 <- rbind(
        merged_df[sample((1:dim(merged_df)[1]), (sample_fraction*dim(merged_df)[1])),] %>%
            ungroup() %>% filter(Category == "True Positive") %>% tukey_hsd(Count ~ test_group) %>% 
            mutate(
                p.adj.signif_2 = case_when(p.adj.signif == "ns" ~ "ns", p.adj.signif != "ns" ~ "sig"),
                pval_2 = p.adj
            ),
        merged_df %>% 
            ungroup() %>% filter(Category == "True Negative") %>% tukey_hsd(Count ~ test_group) %>% 
            mutate(
                p.adj.signif_2 = case_when(p.adj.signif == "ns" ~ "ns", p.adj.signif != "ns" ~ "sig"),
                pval_2 = p.adj
            ),
        merged_df %>% 
            ungroup() %>% filter(Category == "False Positive") %>% tukey_hsd(Count ~ test_group) %>% 
            mutate(
                p.adj.signif_2 = case_when(p.adj.signif == "ns" ~ "ns", p.adj.signif != "ns" ~ "sig"),
                pval_2 = p.adj
            ),
        merged_df %>% 
            ungroup() %>% filter(Category == "False Negative") %>% tukey_hsd(Count ~ test_group) %>% 
            mutate(
                p.adj.signif_2 = case_when(p.adj.signif == "ns" ~ "ns", p.adj.signif != "ns" ~ "sig"),
                pval_2 = p.adj
            )
    )
    set.seed(abs(round(rnorm(1)*100,0)))
    merged_df_stats_subset_3 <- rbind(
        merged_df[sample((1:dim(merged_df)[1]), (sample_fraction*dim(merged_df)[1])),] %>%
            ungroup() %>% filter(Category == "True Positive") %>% tukey_hsd(Count ~ test_group) %>% 
            mutate(
                p.adj.signif_3 = case_when(p.adj.signif == "ns" ~ "ns", p.adj.signif != "ns" ~ "sig"),
                pval_3 = p.adj
            ),
        merged_df %>% 
            ungroup() %>% filter(Category == "True Negative") %>% tukey_hsd(Count ~ test_group) %>% 
            mutate(
                p.adj.signif_3 = case_when(p.adj.signif == "ns" ~ "ns", p.adj.signif != "ns" ~ "sig"),
                pval_3 = p.adj
            ),
        merged_df %>% 
            ungroup() %>% filter(Category == "False Positive") %>% tukey_hsd(Count ~ test_group) %>% 
            mutate(
                p.adj.signif_3 = case_when(p.adj.signif == "ns" ~ "ns", p.adj.signif != "ns" ~ "sig"),
                pval_3 = p.adj
            ),
        merged_df %>% 
            ungroup() %>% filter(Category == "False Negative") %>% tukey_hsd(Count ~ test_group) %>% 
            mutate(
                p.adj.signif_3 = case_when(p.adj.signif == "ns" ~ "ns", p.adj.signif != "ns" ~ "sig"),
                pval_3 = p.adj
            )
    )

    rbind(
        merged_df %>% ungroup() %>% filter(Category == "True Positive") %>%
        tukey_hsd(Count ~ test_group),
        merged_df %>% ungroup() %>% filter(Category == "True Negative") %>%
        tukey_hsd(Count ~ test_group),
        merged_df %>% ungroup() %>% filter(Category == "False Positive") %>%
        tukey_hsd(Count ~ test_group),
        merged_df %>% ungroup() %>% filter(Category == "False Negative") %>%
        tukey_hsd(Count ~ test_group)
    ) -> merged_df_stats_no_groups


    merged_df_stats_subsets <- left_join(
        left_join(
            select(merged_df_stats_no_groups, group1, group2, p.adj.signif, p.adj),
            select(merged_df_stats_subset_1, group1, group2, p.adj.signif_1, pval_1)
        ),
        left_join(
            select(merged_df_stats_subset_2, group1, group2, p.adj.signif_2, pval_2),
            select(merged_df_stats_subset_3, group1, group2, p.adj.signif_3, pval_3)
        )
    )
    merged_df_stats_subsets %>%
        mutate(p.adj = case_when(
            p.adj.signif_1 == "sig" & p.adj.signif_2 == "sig" & p.adj.signif_3 == "sig" & p.adj.signif != "ns" ~ 0.01,
            TRUE ~ 1)) -> merged_df_stats_subsets
    ## TBD
    merged_df_stats_subsets$Category <- "TBD"
    merged_df_stats_subsets$Category[grep("True Positive", merged_df_stats_subsets$group1)] <- "True Positive"
    merged_df_stats_subsets$Category[grep("False Positive", merged_df_stats_subsets$group1)] <- "False Positive"
    merged_df_stats_subsets$Category[grep("True Negative", merged_df_stats_subsets$group1)] <- "True Negative"
    merged_df_stats_subsets$Category[grep("False Negative", merged_df_stats_subsets$group1)] <- "False Negative"
    
    merged_df_stats_w_subsets <- rbind(
        merged_df_stats_subsets %>% ungroup() %>% filter(Category == "True Positive") %>% pGroups(),
        merged_df_stats_subsets %>% ungroup() %>% filter(Category == "False Positive") %>% pGroups(),
        merged_df_stats_subsets %>% ungroup() %>% filter(Category == "True Negative") %>% pGroups(),
        merged_df_stats_subsets %>% ungroup() %>% filter(Category == "False Negative") %>% pGroups()
    )

    ########################################################################

    merged_df %>%
        group_by(Model, Prompt, Category, test_group) %>%
        summarize(mean_count = mean(Count), sd_count = sd(Count)) %>%
        left_join(merged_df_stats_w_subsets, by = c("test_group" = "treatment")) -> merged_df

        ggplot(merged_df) +
            geom_col(aes(x = mean_count, y = Model, fill = Category), color = "black") +
            geom_text(aes(x = mean_count+sd_count+3, y = Model, label = group), color = "black", hjust = 0) +
            geom_errorbar(
                aes(xmin = mean_count-sd_count, xmax = mean_count+sd_count, y = Model, fill = Category),
                color = "black", width = 0.5
            ) +
            facet_grid(Prompt~Category, scales = "free_y") +
            scale_x_continuous(
                breaks = seq(0,100,25), limits = c(0,110),
                name = "Count (number of articles)"
            ) +
            scale_y_discrete(name = "") +
            scale_fill_manual(values = c("#d7191c", "#fdae61", "#a6d96a", "#1a9641"), guide = "none") +
            theme_bw() +
            theme(
                strip.text.y = element_text(angle = 0),
                text = element_text(size = 16)
            ) -> results_plot

    # Precision and recall
    merged_df_raw_pr <- merged_df_raw2 %>% ungroup() %>% select(Category, Count, Model, Prompt, Replicate) %>%
        pivot_wider(names_from = "Category", values_from = "Count", values_fill = 0) %>%
        rename_with(~ gsub(" ", "_", tolower(.x)), starts_with("False") | starts_with("True")) %>% 
        mutate(
            precision = true_positive / (true_positive + false_positive),
            recall = true_positive / (true_positive + false_negative),
            f1_score = 2 * (precision * recall) / (precision + recall),
            overall_accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        )
    # merged_df_raw_pr[which.min(merged_df_raw_pr$overall_accuracy),]
    # merged_df_raw_pr[which.max(merged_df_raw_pr$overall_accuracy),]

    merged_df_raw_pr %>%
        ggplot() +
            geom_point(
                aes(y = precision, x = recall, fill = Model, shape = Prompt),
                size = 4, alpha = 0.8
            ) +
            
            scale_shape_manual(values = c(21:23)) +
            scale_fill_brewer(palette = "Set1") +
            guides(fill = guide_legend(override.aes = list(shape = 21))) +
            scale_y_continuous(name = "Recall") +
            scale_x_continuous(name = "Precision") +
            theme_bw() +
            theme(
                # legend.position = "left"
            ) -> pr_plot

    merged_df_raw_pr %>%
        group_by(Prompt, Model) %>%
        summarize(f1_score = mean(f1_score)) %>%
        ggplot(aes(x = Prompt, y = Model, fill = f1_score)) +
            geom_tile(color = "black") +
            geom_text(aes(label = round(f1_score,2)), color = "white") +
            scale_fill_gradient(low = "grey", high = "black", "F1 Score") +
            theme_bw() +
            scale_y_discrete(position = "right", name = "") +
            theme() -> f1_plot

## Merge Zem
    aspect = 1
    pdf(
        file = "/project_data/shared/ai_in_phytochemistry/figures/proteins.pdf",
        width = 16 * aspect, height = 8 * aspect
    )
    
        plot_grid(
            process_image, 
            plot_grid(
                results_plot, 
                plot_grid(
                    pr_plot, f1_plot, align = "v", axis = "lr",
                    ncol = 1, rel_heights = c(2,1), labels = c("", "D")
                ),
                labels = c("", "C"), nrow = 1, rel_widths = c(2,1)
            ),
            ncol = 1, labels = c("A", "B"), rel_heights = c(1.5,2)
        )
    
    invisible(dev.off())
    print(magick::image_read_pdf("/project_data/shared/ai_in_phytochemistry/figures/proteins.pdf"), info=F)

two1.0\textbf{Workflow and results of protein-product relationship evaluation.} \textbf{A} Workflow used to generate results. Search terms indicate enzymes searched for in NCBI (for example, "beta-amyrin synthase" or "cycloartenol synthase"). Peer-reviewed literature indicates the manually curated database of NCBI protein entries that describe verified protein-product pairs. The collection is made up of both the products of the search terms and the contents of the curated database. Blue lines indicate NCBI protein records, yellow lines indicate cells created through manual evaluation, black boxes represent cells generated via language model response and comparison between language model response and manual evaluation. \textbf{B} Outcomes of language model classification of protein-product pairs. The y-axis represents the different models used, the x-axis represents the number of entries evaluated. Red indicates false negative, orange denotes false positive, light green signifies true negatives, and dark green indicates true positives. In this case, a true positive indicates a record the language model correctly labeled as a positive, a true negative indicates a record the model correctly labeled as a negative, a false positive is a record the model erroneously labeled as a positive, and a false negative a record that the model erroneously labeled as a negative. Bar heights and error bars indicate the mean and standard deviation of n = 3 replicates, respectively. In some cases, a model-prompt pair generated the same outcome in all three replicates, leading to standard deviation values of zero and thus appearing in this figure as a bar without an error bar. {fig:rachel}
cat(paste0(
    "NoVerbatim",
    " The goal in this subsection was to assess the ability of large language models to distinguish between database",
    " records that report the validated product of an enzyme catalyst (positive entries) versus those that report a",
    " predicted product (negative entries). The first step towards this goal was to obtain a collection of database records",
    " with both positive and negative representatives. To obtain this collection we first conducted command-line NCBI searches",
    " using three enzyme-related search terms (\"beta-amyrin synthase\", \"lupeol synthase\", \"cycloartenol synthase\")",
    " and added the first 20 records from each search into our collection, resulting in an initial set of 60 records",
    " identified by their protein IDs (Fig. \\ref{fig:rachel}A). Manual inspection of these 60 records revealed an imbalance,",
    " with a predominance of negative records (reports of enzymes and their predicted, not validated, products).",
    " To create a balanced collection, we supplemented the initial set of 60 with additional records that we identified based on",
    " manual inspection of peer-reviewed articles that reported the validated products of specific enzymes.",
    " After these steps our collection consisted of ",
    dim(correct_answers)[1],
    " records, including ",
    dim(filter(correct_answers, Manual == "Yes"))[1], " positive records and ",
    dim(filter(correct_answers, Manual == "No"))[1], " negative records.\n\n",

    "With a collection of manually validated positive and negative records in hand, we next retrieved the full contents of each record,",
    " including the PubMed IDs of any articles describing the enzyme in the record. In a separate step, we retrieved the",
    " abstracts of the articles with those PubMed IDs and associated those abstract with the corresponding records in",
    " our collection. Thus, at this stage, our collection consisted of the ",
    dim(correct_answers)[1],
    " records, each with their Protein ID, full record contents, associated abstracts, and manual annotation as a",
    " positive or negative record (Fig. \\ref{fig:rachel}A).",
    " Finally, we looped over all the records in our collection and passed them 1-by-1 to an OpenAI language model alongside",
    " a system prompt designed to elicit a response from the model labelling the record as positive or negative.",
    " We compared the label assigned by the language model against our manually assigned label, and, considering the",
    " manually assigned label to be factual, labeled each language model response as a true positive, true negative,",
    " false positive, or false negative. In this case, a true positive indicates a record the language model correctly",
    " labeled as a positive, a true negative indicates a record the model correctly labeled as a negative,",
    " a false positive is a record the model erroneously labeled as a positive, and a false negative a record that the model",
    " erroneously labeled as a negative.\n\n",

    "Using the approach above, we tested the accuracy of several language models under the instruction of several different styles of system prompts.",
    " A system prompt is a model input that describes a task that the language model should perform.",
    " System prompts can range from very simple to very complex and have a substantial impact on the efficacy of the model \\cite{zhang2024sprig, grabb2023impact, schick2021size}.",
    " Furthermore, through the process of prompt engineering, a prompt can be optimized to increase the system's performance against a given metric \\cite{diab2022stable}.",
    " A key example emphasizing the importance of prompt engineering is the \"chain-of-thought\" prompting technique developed in 2022 \\cite{wei2022chain}.",
    " The most basic form of the method consists of simply adding the words \"Let's think through this step-by-step\" into an existing prompt.",
    " The researchers found that by adding these words to a prompt,",
    " model performance increased substantially (from 18% to 80% accuracy on the MultiArith test using OpenAI GPT3) \\cite{wei2022chain}.",
    " The chain-of-thought prompting method has since become a standard prompting technique for increasing language model performance,",
    " to the extent that OpenAI has incorporated it as a default behavior into some of its most recent models \\cite{wang2024planning}.",
    " Here, we evaluated the accuracy of four different language models (gpt-3.5-turbo, gpt-4o-2024-08-06, gpt-4-0125-preview, and gpt-4o-mini) using one of three different system prompts (a standard, non-chain-of-thought prompt;",
    " a \"let's think step-by-step\" chain-of-thought prompt, and a more detailed chain-of-thought prompt that asked the model to include intermediate steps in its output).",
    " Details of all system prompts used in this section are provided in Supplemental File 1,",
    " but all prompts were geared toward asking the model \"Does the enzyme described in the record make the product?\"."
    
))


## Image of process
    squeeze <- 0
    process_image <- ggplot(data = data.frame(x = c(0,1), y = c(0.5,0.5))) +
    geom_point(aes(x = x, y = y), color = "white") +
    theme_void() +
    scale_y_continuous(limits = c(0,1)) +
    annotation_custom(
        rasterGrob(
            readPNG(
                "/project_data/shared/ai_in_phytochemistry/figures/rag_diagram2.png"
            ), interpolate=TRUE
        ), xmin=0, xmax=1, ymin=0+squeeze, ymax=1-squeeze
    )

################################################
################## WAND B ######################
################################################

data <- read_csv("/project_data/shared/ai_in_phytochemistry/wb_rag_data/wandb-raw.csv")

## Performance histogram
    data %>%
      ggplot(aes(x = Percentage)) +
        geom_histogram(aes(fill = Percentage, group = Percentage), color = "black", binwidth = 5) +
        scale_fill_gradient2(
            low = "#f46d43", mid = "#ffffbf", high = "#66c2a5", name = "Prompt Performance (% Correct)",
            midpoint = 77.5, # Adjust this based on your mid value (e.g., mean or median of the data)
            limits = c(65, 90),
            guide = "none"#
        ) +
        scale_x_continuous(
            name = "Prompt Performance\n(% Correct)", breaks = seq(65,90,5)
        ) +
        scale_y_continuous(
            name = "Prompts Generated",
            position = "right"
        ) +
        theme_minimal() +
        coord_flip() +
        theme(
            legend.position = c(0.8, 0.5),
            plot.margin = margin(0,0,0,0)
        ) -> prompt_histogram_plot

## PCA and kmeans
    # set.seed(334)
    # n <- abs(round(rnorm(1), 3)*1000)
    # n
    # set.seed(n)
    set.seed(491)
    
    runMatrixAnalysis(
      data = data,
      analysis = "pca",
      columns_w_values_for_single_analyte = colnames(data)[grep("embed", colnames(data))],
      columns_w_sample_ID_info = c("prompt_number", "Percentage", "Prompt")
    ) %>%
      arrange(Percentage) -> pca_output
    runMatrixAnalysis(
      data = pca_output,
      analysis = "kmeans",
      parameters = 7,
      columns_w_values_for_single_analyte = c("Dim.1", "Dim.2"),
      columns_w_sample_ID_info = c("prompt_number", "Percentage", "Prompt")
    ) -> pca_output
    pca_output$cluster <- factor(pca_output$cluster)

    pca_output$sil <- as.numeric(silhouette(
        as.numeric(unlist(select(pca_output, cluster))),
        runMatrixAnalysis(
          data = pca_output,
          analysis = "dist",
          columns_w_values_for_single_analyte = c("Dim.1", "Dim.2"),
          columns_w_sample_ID_info = c("prompt_number", "Percentage")
        )
    )[,"sil_width"])

    pca_output_summarized <- group_by(pca_output, cluster) %>% summarize(mean = mean(sil), sd = sd(sil))
      ggplot() +
        geom_col(data = pca_output_summarized, aes(x = cluster, y = mean), color = "black", fill = "white") +
        geom_jitter(data = pca_output, aes(x = cluster, y = sil), width = 0.2, shape = 21, color = "black", fill = "white") +
        geom_errorbar(data = pca_output_summarized, aes(x = cluster, y = mean, ymax = mean+sd, ymin = mean-sd), width = 0.5) +        
        geom_hline(data = data.frame(y = 0.2), aes(yintercept = y), linetype = "dashed") +
        scale_y_continuous(name = "Silhouette\nScore") +
        scale_x_discrete(name = "Cluster") +
        theme_classic() -> cluster_silhouette_barplot

    range_multiplier = 1.5
    pca_output %>%
      ggplot() +
        geom_plot(
            data = tibble(
                x = (min(pca_output$Dim.1)-3), y = 40*range_multiplier,
                # x = (max(pca_output$Dim.1)+1), y = 36*range_multiplier,
                plot = list(prompt_histogram_plot)
            ),
            aes(x, y, label = plot), vp.width = 0.5, vp.height = 0.3
        ) +
        geom_mark_ellipse(
          aes(x = Dim.1, y = Dim.2, group = cluster, label = cluster),
          size = 0.5, alpha = 0.9
        ) +
        geom_point(
          aes(x = Dim.1, y = Dim.2, fill = Percentage, size = abs(sil)),
          shape = 21, alpha = 1#size = 5
        ) +
        scale_fill_gradient2(
            low = "#f46d43", mid = "#ffffbf", high = "#66c2a5",
            name = "Prompt Performance\n(% Correct)",
            midpoint = 77.5, # Adjust this based on your mid value (e.g., mean or median of the data)
            limits = c(65, 90) #
        ) +
        scale_size(guide = "none") +
        scale_x_continuous(
            limits = c(
                (min(pca_output$Dim.1)-3),
                (max(pca_output$Dim.1)+3)
            ),
            name = "Dimension 1",
            expand = c(0,0)
        ) +
        scale_y_continuous(
            limits = c(-25*range_multiplier,40*range_multiplier),
            name = "Dimension 2",
            expand = c(0,0)
        ) +
        guides(
            fill = guide_legend(
                title.position = "top", title.hjust=0.5,
                override.aes = list(size = 5)
            )
        ) +
        theme_minimal() +
        theme(
            legend.position = c(0.8, 0.9),
            legend.direction = "horizontal"
        ) -> prompt_pca_plot
   
    pgroup_data <- dunnTest(pca_output, Percentage ~ cluster) %>% pGroups()
    pca_output_summarized <- group_by(pca_output, cluster) %>% summarize(mean = mean(Percentage), sd = sd(Percentage))
      ggplot() +
        geom_col(data = pca_output_summarized, aes(x = cluster, y = mean), color = "black", fill = "white") +
        geom_jitter(data = pca_output, aes(x = cluster, y = Percentage), width = 0.2, shape = 21, color = "black", fill = "white") +        
        geom_errorbar(data = pca_output_summarized, aes(x = cluster, y = mean, ymax = mean+sd, ymin = mean-sd), width = 0.5) +
        geom_text(data = pgroup_data, aes(x = treatment, y = 100, label = group)) +
        scale_y_continuous(expand = c(0,0), limits = c(0,105), name = "Prompt\nPerformance\n(% Correct)") +
        scale_x_discrete(name = "Cluster") +
        theme_classic() -> cluster_performance_barplot

#############################################
################## RAG ######################
#############################################

terms <- c(
    "assistant", "you_are_an_ai_classifier", "say_true_or_false",
    "beware_of_synonyms", "use_background_knowledge",
    "youre_going_to_want_to_say_false", "go_ahead_and_say_true",
    "~abstract", "~contents"
)
results_1 <- read_csv("/project_data/shared/ai_in_phytochemistry/vingette_2_RAG/rag_output_processed_rep1.csv", show_col_types = FALSE)
results_2 <- read_csv("/project_data/shared/ai_in_phytochemistry/vingette_2_RAG/rag_output_processed_rep2.csv", show_col_types = FALSE)
results_3 <- read_csv("/project_data/shared/ai_in_phytochemistry/vingette_2_RAG/rag_output_processed_rep3.csv", show_col_types = FALSE)
results <- rbind(
    cbind(data.frame(replicate = 1), results_1),
    cbind(data.frame(replicate = 2), results_2),
    cbind(data.frame(replicate = 3), results_3)
)
results <- results[,-c(grep("content_used", colnames(results)))]
colnames(results) <- gsub("gpt_response_", "", colnames(results))
results <- pivot_longer(
    results, cols = grep(paste(terms, collapse = "|"), colnames(results), value = TRUE),
    names_to = "prompt_combo", values_to = "gpt_response"
)

results$gpt_response <- toupper(substr(as.character(results$gpt_response), 1, 1))
results$manual_evaluation <- toupper(substr(as.character(results$manual_evaluation), 1, 1))

results %>%
    filter(manual_evaluation != "U") %>%
    filter(gpt_response %in% c("T", "F")) %>%
    mutate(judgement = case_when(
        gpt_response == "T" & manual_evaluation == "F" ~ "false_positive",
        gpt_response == "F" & manual_evaluation == "T" ~ "false_negative",
        gpt_response == manual_evaluation & gpt_response == "T" ~ "true_positive",
        gpt_response == manual_evaluation & gpt_response == "F" ~ "true_negative",
        TRUE ~ "unknown"
    )) -> results
results$status <- gsub("_.*$", "", results$judgement)
results$category <- gsub(".*_", "", results$judgement)

results <- results %>%
    group_by(replicate, prompt_combo, judgement, status, category) %>%
    summarize(count = n()) %>%
    ungroup() %>% group_by(prompt_combo, replicate) %>%
    mutate(percent = count/sum(count)*100) %>%
    group_by(prompt_combo, judgement, status, category) %>%
    summarize(mean_percent = mean(percent), sd_percent = sd(percent), percent = percent)

for (term in terms) {
    results[[term]] <- FALSE
    results[[term]][grep(term, results$prompt_combo)] <- TRUE
}

results <- results %>%
    pivot_longer(
        cols = terms,
        names_to = "prompt_characteristic",
        values_to = "presence"
    ) %>%
    filter(presence == TRUE) %>%
    group_by(prompt_combo, judgement) %>%
    reframe(
        prompt_combination = list(unique(prompt_characteristic)),
        percent = percent,
        mean_percent = mean_percent,
        sd_percent = sd_percent
    ) %>% ungroup()
# str(results)
results$status <- gsub("_.*$", "", results$judgement)
results$category <- gsub(".*_", "", results$judgement)

rbind(
    results %>% ungroup() %>% filter(judgement == "true_positive") %>%
    # tukey_hsd(percent ~ prompt_combo) %>% pGroups() %>%
    dunnTest(percent ~ prompt_combo) %>% pGroups() %>%
    mutate(judgement = "true_positive"),
    results %>% ungroup() %>% filter(judgement == "true_negative") %>%
    # tukey_hsd(percent ~ prompt_combo) %>% pGroups() %>%
    dunnTest(percent ~ prompt_combo) %>% pGroups() %>%
    mutate(judgement = "true_negative"),
    results %>% ungroup() %>% filter(judgement == "false_positive") %>%
    # tukey_hsd(percent ~ prompt_combo) %>% pGroups() %>%
    dunnTest(percent ~ prompt_combo) %>% pGroups() %>%
    mutate(judgement = "false_positive"),
    results %>% ungroup() %>% filter(judgement == "false_negative") %>%
    # tukey_hsd(percent ~ prompt_combo) %>% pGroups() %>%
    dunnTest(percent ~ prompt_combo) %>% pGroups() %>%
    mutate(judgement = "false_negative")
) -> results_stats

results <- left_join(results, results_stats, by = c("prompt_combo" = "treatment", "judgement" = "judgement"))

# Plot with adjusted error bars
ggplot() +
	geom_col(
		data = unique(select(results, prompt_combination, mean_percent, judgement)),
		aes(x = prompt_combination, y = mean_percent, fill = judgement),
		color = "black"
	) +
    geom_text(
        data = unique(select(results, prompt_combination, group, mean_percent, sd_percent, judgement)),
        aes(x = prompt_combination, y = mean_percent+(sd_percent+12), label = group), size = 3, vjust = 1
    ) +
	geom_errorbar(
		data = unique(select(results, prompt_combination, mean_percent, sd_percent, judgement)),
		aes(
			x = prompt_combination, 
			ymin = mean_percent+sd_percent,
			ymax = mean_percent-sd_percent
		),
		color = "black", width = 0.25
	) +
    facet_grid(judgement~.) +
	scale_x_upset(name = "Prompt Features", order_by = "degree") +
	scale_y_continuous(
		name = "Proportion of Articles (%)",
		breaks = seq(-100, 100, 10), labels = abs(seq(-100, 100, 10))
	) +
	scale_fill_manual(
		values = c("#9c2700", "#e00025", "#2dd413", "#1B9E77"),
		name = ""
	) +
	theme_bw() +
	theme(
		legend.position = "top",
        strip.text = element_blank()
	) +
	guides(fill = guide_legend(nrow = 2)) -> rag_results

################################################
################## MERGE ZEM ###################
################################################

aspect = 1.5
pdf(
    file = "/project_data/shared/ai_in_phytochemistry/figures/rag.pdf",
    width = 4 * aspect, height = 9 * aspect
)
    plot_grid(
        prompt_pca_plot,
        plot_grid(
            cluster_silhouette_barplot, cluster_performance_barplot,
            nrow = 1, align = "h", axis = "tb", labels = c("B", "C")
        ),
        plot_grid(
            process_image, rag_results, nrow = 1, rel_widths = c(1,1),
            labels = c("D", "E")
        ),
        ncol = 1, rel_heights = c(2, 0.6, 2), labels = "A"
    )
invisible(dev.off())
print(magick::image_read_pdf("/project_data/shared/ai_in_phytochemistry/figures/rag.pdf"), info=F)
one0.9\textbf{Automated prompt engineering and retrieval-augmented generation to enhance the identification of compound-species associations.} \textbf{A} Performance (inset histogram) and principal components analysis of the linguistic meaning of 50 prompts generated for the task of compound-species relationship identification. Ellipses indicate clusters of linguistically similar prompts. \textbf{B} Silhouette scores for each point in each cluster, where scores > 0.5 suggest strong cluster membership, scores between 0.2 and 0.5 indicate moderate support, and scores below 0.2 show low cluster association. \textbf{C} Performance of the prompts in each cluster. Bar heights and error bars indicate means and standard deviations of n = 3 replicates, respectively. Letters indicate statistical groups determined by Tukey's HSD test (adjusted p < 0.05). \textbf{D} Model of retrieval-augmented generation. \textbf{E} Performance of different system prompts in combination with retrieval-augmented generation. Contents of the system prompts are indicated on the x-axis. Colors indicate true/false positives/negatives, as described for Figure 1. {fig:braidon}

## Image of process
    process_image <- ggplot(data = data.frame(x = c(0,1), y = c(0.5,0.5))) +
    geom_point(aes(x = x, y = y), color = "white") +
    theme_void() +
    annotation_custom(
        rasterGrob(
            readPNG(
                "/project_data/shared/ai_in_phytochemistry/figures/flow.png"
            ), interpolate=TRUE
        ), xmin=0, xmax=1, ymin=0, ymax=1
    )

## Read in and process tables
    input_directory <- "/project_data/shared/ai_in_phytochemistry/vingette_3_visionAI/articles/phytosterols/png_annot/"
    csv_files <- dir(input_directory)[grep("_extracted_processed.csv", dir(input_directory))]
    csv_files <- paste0(input_directory, csv_files)
    output <- list()
    for(i in 1:length(csv_files)) { # i=2
        
        ## Read in, fix NaN, fix compound_name column
        data <- read_csv(csv_files[i], show_col_types = FALSE)
        print(dim(data))
        print(dim(data)[1]*dim(data)[2])
        colnames(data)[1] <- "compound_name"
        # Loop over columns (except the first, if it's a name column like "Phytosterols") and convert to numeric
        data[,-1] <- suppressWarnings(lapply(data[,-1], function(x) as.numeric(as.character(x)))) 
        # Remove rows where all non-first columns contain only NA (resulting from non-numeric values)
        data <- data[rowSums(is.na(data[,-1])) != ncol(data[,-1]),] 
        
        ## Merge duplicate compound_names
        data <- pivot_longer(data, cols = 2:dim(data)[2], names_to = "genus_species", values_to = "abundance")
        data$abundance[is.na(data$abundance)] <- 0
        data %>%
            group_by(genus_species) %>%
            mutate(abundance = abundance/sum(abundance)*100) %>%
            group_by(compound_name, genus_species) %>%
            summarize(abundance = sum(abundance)) -> data
        data$abundance[is.na(data$abundance)] <- 0
        empties <- group_by(data, genus_species) %>% summarize(tot = sum(abundance))
        empties <- filter(empties, tot == 0)$genus_species
        data <- filter(data, !genus_species %in% empties)
        data$source <- gsub("_extracted_processed.csv", "", gsub(".*/", "", csv_files[i]))
        output[[i]] <- data
    }
    data <- do.call(rbind, output)
    
## Clean up names
    data$genus_species <- gsub("\\.", "", data$genus_species)
    data$compound_name <- tolower(data$compound_name)
    data$compound_name <- gsub("sitosterol", "b-sitosterol", data$compound_name)
    data$compound_name <- gsub("b-b-", "b-", data$compound_name)
    data$compound_name <- gsub("α-", "a-", data$compound_name)
    data$compound_name <- gsub("β-", "b-", data$compound_name)
    data$compound_name <- gsub("β ", "", data$compound_name)
    data$compound_name <- gsub("δ", "d", data$compound_name)
    data$compound_name <- gsub("δ-", "d", data$compound_name)
    data$compound_name <- gsub("b-b-", "b-", data$compound_name)
    data$sample_unique_ID <- paste0(data$genus_species, "_", data$source)

## Accuracy piechart
    data.frame(
        category = c("correct", "incorrect"),
        percent = c(98, 2)
    ) %>%
        ggplot(aes(x = 1, y = percent, fill = category)) +
            geom_col(color = "black") +
            coord_polar(theta = "y") +
            scale_fill_manual(values = c("#66c2a5", "#f46d43")) +
            theme_void() -> pie_chart

## Presence absence analysis
    presence_absence_df <- data %>%
        mutate(presence = abundance > 0) %>%
        group_by(genus_species, compound_name) %>%
        summarize(presence = any(presence)) %>%
        pivot_wider(names_from = genus_species, values_from = presence, values_fill = FALSE)
    
    presence_absence_df <- data %>%
        mutate(presence = ifelse(abundance > 0, 1, 0)) %>%
        group_by(genus_species, compound_name) %>%
        summarize(presence = max(presence)) %>%
        pivot_wider(names_from = genus_species, values_from = presence, values_fill = 0)

    data %>%
        filter(abundance > 0) %>%
        select(genus_species, compound_name) %>%
        distinct() %>%
        group_by(genus_species) %>%
        summarize(n_compound_reported = n()) -> n_compound_reported
    ggplot(n_compound_reported) +
        geom_histogram(aes(x = n_compound_reported), color = "black", fill = "grey") +
        theme_bw() -> n_compound_reported_plot
    
    data <- pivot_wider(select(data, compound_name, sample_unique_ID, abundance), names_from = "sample_unique_ID", values_from = "abundance", values_fill = NA)
    data <- pivot_longer(data, names_to = "sample_unique_ID", values_to = "abundance", cols = c(2:(dim(data)[2])))
    data$source <- gsub(".*_", "", data$sample_unique_ID)
    data$genus_species <- gsub("_.*$", "", data$sample_unique_ID)
    
    data %>%
        group_by(compound_name) %>%
        summarize(compound_abu = mean(abundance, na.rm = TRUE)) -> compound_abu
    ggplot(compound_abu) +
        geom_histogram(aes(x = compound_abu), color = "black", fill = "grey") +
        theme_bw() -> compound_abu_histogram_plot

## Build tree, plot tree and heat map
    data$genus_species <- gsub(" ", "_", data$genus_species)
    fortify(buildTree(
        scaffold_type = "newick",
        scaffold_in_path = "/project_data/shared/general_lab_resources/phylogenies/angiosperms.newick",
        members = unique(data$genus_species)
    )) -> tree

#### HERE MERGE THE TREE AND THE DATA AND MAKE THE HEAT MAP FROM THE MERGE!!
                                             
    data %>%
        filter(compound_name %in% filter(compound_abu, compound_abu > 5)$compound_name) %>%
        left_join(select(tree, label, y), by = c("genus_species" = "label")) %>%
        filter(abundance != "NA") -> data

        data <- filter(data, genus_species %in% unique(filter(tree, isTip == TRUE)$label))
        # print(paste0("n tips ", length(unique(filter(tree, isTip == TRUE)$label))))
        x_labels <- unlist(select(drop_na(arrange(unique(select(ungroup(data), y, genus_species)), y)), genus_species))
        # print(paste0("n heatmap entries ", length(unique(data$genus_species))))

        ggplot(data) +
            geom_tile(aes(y = compound_name, x = y, fill = abundance), color = "black") +
            scale_fill_gradient(low = "#ffffbf", high = "#d53e4f", name = "Rel.\nabundance") +
            scale_x_continuous(
                limits = c(0,(length(x_labels)+1)), expand = c(0,0),
                breaks = seq(1,length(x_labels), 1),
                labels = x_labels, name = "", position = "top"
            ) +
            scale_y_discrete(name = "") +
            # facet_grid(.~source, scales = "free", space = "free") +
            theme_bw() +
            theme(
                axis.text.x = element_blank(),
                axis.title.x = element_blank(),
                # axis.text.x = element_text(angle = 90, hjust = 0.5, vjust = 0.5),
                plot.margin = margin(-20,0,0,0)
            ) -> heatmap_plot

# data
# length(unique(filter(tree, isTip == TRUE)$label))
                                             
## Apiaceae stat test
    descendantsFortified(tree) %>%
        filter(parent == 122) %>% select(label) %>% unlist() -> api
    api[nchar(api) > 4] -> api
    api <- gsub("_", " ", api)
    data %>%
        filter(compound_name %in% c("stigmasterol", "b-sitosterol")) %>%
        select(sample_unique_ID, compound_name, abundance) %>%
        pivot_wider(names_from = "compound_name", values_from = "abundance") %>%
        filter(`b-sitosterol` > 0 & stigmasterol > 0) %>%
        # mutate(ratio = `b-sitosterol`/stigmasterol) %>% drop_na() %>% 
        mutate(ratio = stigmasterol/`b-sitosterol`) %>% drop_na() %>% 
        mutate(genus_species = gsub("_.*$", "", sample_unique_ID)) %>%
        mutate(api = case_when(genus_species %in% api ~ "Apiaceous", ! genus_species %in% api ~ "non-\nApiaceous")) -> api
    api %>% wilcoxTest(ratio ~ api) -> api_test

    api_test_annot <- data.frame(
        ymin = api_test$group1,
        ymax = api_test$group2,
        x_position = c(2.5),
        text = "**",
        text_size = 20,
        text_vert_offset = 1.5,
        text_horiz_offset = 0.25,
        tip_length_ymin = 0.3,
        tip_length_ymax = 0.8,
        hjust = 0.5,
        vjust = 0.5
    )

    ggplot(api) +
        geom_jitter(
            aes(x = ratio, y = api, fill = ratio),
            height = 0.2, width = 0, color = "black", alpha = 0.6, size = 4, shape = 21
        ) +
        geomSignif(data = api_test_annot, orientation = "vertical") +
        scale_y_discrete(name = "") +
        scale_fill_gradient(low = "#ffffbf", high = "#d53e4f", name = "Ratio") +
        scale_x_continuous(name = "Ratio of b-sitosterol to stigmasterol") +
        theme_bw() -> api_stat_test


### Plot the tree
    tree <- left_join(tree, unique(select(ungroup(data), genus_species, source)), by = c("label" = "genus_species"))
    tree %>% group_by(label) %>% mutate(source = n()) %>% unique() -> tree
    ggtree(tree) +
        # geom_label(aes(x = x, y = y, label = node))
        geom_tiplab(
            angle = 90, vjust = 0.5, hjust = 0,
            align = TRUE, offset = -350, geom = "text",
            aes(label = y)
        ) +
        ggtext::geom_richtext(
            data = filter(tree, isTip == TRUE),
            aes(x = 810, y = y, label = label),
            angle = 90, label.size = 0, hjust = 0,
            color = "black", label.colour = "white", size = 4
        ) +
                                             
        geom_point(
            data = filter(tree, isTip == TRUE),
            aes(x = x+40, y = y),
            shape = 21, color = "black", fill = "white", size = 5
        ) +
        geom_text(
            data = filter(tree, isTip == TRUE),
            aes(x = x+40, y = y, label = source), size = 4
        ) +
        geom_plot(
            data = tibble(
                x = 5, y = 25,
                # x = (max(pca_output$Dim.1)+1), y = 36*range_multiplier,
                plot = list(api_stat_test)
            ),
            aes(x, y, label = plot), vp.width = 0.6, vp.height = 0.25
        ) +
        scale_fill_manual(values = discrete_palette, name = "Source") +
        coord_flip() +
        guides(fill = guide_legend(ncol = 4)) +
        scale_x_reverse(limits = c(820,0)) +
        scale_y_continuous(limits = c(0,(length(unique(x_labels))+1)), expand = c(0,0)) +
        geom_treescale(x = -100, y = 5, width = 50, offset = 1) +
        theme_void() +
        theme(
            plot.margin = margin(10,0,0,0),
            legend.position = c(0.6, 0.9)
        ) -> tree_plot

## Merge zem
    aspect = 1
    pdf(
        file = "/project_data/shared/ai_in_phytochemistry/figures/vision.pdf",
        width = 16 * aspect, height = 10 * aspect
    )
            # suppressWarnings(
            #     plot_grid(
            #         process_image,
            #         plot_grid(
            #             pie_chart, n_compound_reported_plot, compound_abu_histogram_plot, api_stat_test,
            #             labels = c("B", "C", "D", "F"), nrow = 1, rel_widths = c(1,2,2,2)
            #         ),
            #         plot_grid(tree_plot, heatmap_plot, ncol = 1, align = "v", axis = "lr", rel_heights = c(4,2)),
            #         ncol = 1, rel_heights = c(1,1,3.2), labels = c("A", "", "E")
            #     )
            # )
            suppressWarnings(
                plot_grid(
                    plot_grid(process_image,pie_chart, rel_widths = c(6,1), labels = c("A", "B")),
                    # plot_grid(
                    #     , n_compound_reported_plot, compound_abu_histogram_plot, api_stat_test,
                    #     labels = c("B", "C", "D", "F"), nrow = 1, rel_widths = c(1,2,2,2)
                    # ),
                    plot_grid(tree_plot, heatmap_plot, ncol = 1, align = "v", axis = "lr", rel_heights = c(4,2)),
                    ncol = 1, rel_heights = c(1,3.4), labels = c("", "C")
                )
            )
    invisible(dev.off())
    print(magick::image_read_pdf("/project_data/shared/ai_in_phytochemistry/figures/vision.pdf"), info=F)
two1.0\textbf{Workflow and results from using a multimodal language model to transcribe images of scientific tables.} \textbf{A} Workflow used to transcribe/extract and processes text from images of tables. \textbf{B} Pie chart indicating the accuracy of the transcription process based on manual comparison of transcribed data and the original tables. \textbf{C} Phylochemical map showing the abundance (in heat map, abundance indicated in legend) of different phytosterols across the species included in this study (in phylogenetic tree). The phylogeny was prepared by pruning a previously published megaphylogeny \cite{Qian_2015}. The inset plot shows the ratio of beta-sitosterol to stigmasterol in Apiaceous versus non-Apiaceous species. Stars indicate statistically significant differences in the mean ratio between the two groups (Wilconxon test, p < 0.05). {fig:luke}
## Run this chunk to compile the preprint
source("/project_data/shared/general_lab_resources/preprints/preprintCompiler.R")
compilePreprint(
    project_directory = "/project_data/shared/ai_in_phytochemistry/",
    notebook_name = "ai_in_phytochemistry",
    path_to_bib_file = "/project_data/shared/general_lab_resources/literature/bustalab.bib"
)
