library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(pROC)
library(corrplot)
library(ggplot2)
library(scales)
library(gridExtra)

set.seed(123)

hotel_data <- read.csv("hotels.csv")

hotel_data <- hotel_data %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(is_canceled = factor(is_canceled))

hotel_data <- hotel_data %>%
  mutate(
    children = ifelse(is.na(children), 0, children),
    country = ifelse(is.na(country), "UNK", as.character(country)),
    agent = ifelse(is.na(agent), "None", as.character(agent)),
    company = ifelse(is.na(company), "None", as.character(company))
  ) %>%
  mutate(
    country = as.factor(country),
    agent = as.factor(agent),
    company = as.factor(company)
  )

hotel_data <- hotel_data %>%
  mutate(
    total_nights = stays_in_weekend_nights + stays_in_week_nights,
    total_guests = adults + children + babies,
    has_children = ifelse(children > 0 | babies > 0, 1, 0),
    booking_changes_binary = ifelse(booking_changes > 0, 1, 0),
    special_requests_binary = ifelse(total_of_special_requests > 0, 1, 0),
    arrival_date_month = factor(arrival_date_month, levels = month.name),
    is_repeated_guest = factor(is_repeated_guest),
    has_children = factor(has_children),
    booking_changes_binary = factor(booking_changes_binary),
    special_requests_binary = factor(special_requests_binary)
  )

selected_features <- c(
  "is_canceled", "lead_time", "arrival_date_month", "arrival_date_year",
  "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
  "meal", "market_segment", "distribution_channel", "is_repeated_guest",
  "previous_cancellations", "previous_bookings_not_canceled", "reserved_room_type",
  "assigned_room_type", "booking_changes", "deposit_type", "days_in_waiting_list",
  "customer_type", "adr", "required_car_parking_spaces", "total_of_special_requests",
  "total_nights", "total_guests", "has_children", "booking_changes_binary", 
  "special_requests_binary"
)

model_data <- hotel_data %>%
  select(all_of(selected_features))

custom_describe <- function(df) {
  numeric_cols <- df %>% select(where(is.numeric)) %>% names()
  
  result <- data.frame(
    variable = numeric_cols,
    n = sapply(df[numeric_cols], function(x) sum(!is.na(x))),
    mean = sapply(df[numeric_cols], mean, na.rm = TRUE),
    sd = sapply(df[numeric_cols], sd, na.rm = TRUE),
    median = sapply(df[numeric_cols], median, na.rm = TRUE),
    min = sapply(df[numeric_cols], min, na.rm = TRUE),
    max = sapply(df[numeric_cols], max, na.rm = TRUE)
  )
  
  return(result)
}

summary_stats <- custom_describe(model_data)
print(summary_stats)

index <- createDataPartition(model_data$is_canceled, p = 0.7, list = FALSE)
train_data <- model_data[index, ]
test_data <- model_data[-index, ]

p1 <- ggplot(train_data, aes(x = is_canceled)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Cancellations", x = "Is Canceled", y = "Count") +
  theme_minimal()
print(p1)

p2 <- ggplot(train_data, aes(x = lead_time, fill = is_canceled)) +
  geom_density(alpha = 0.5) +
  labs(title = "Lead Time Distribution by Cancellation Status", x = "Lead Time (days)", y = "Density") +
  theme_minimal()
print(p2)

p3 <- train_data %>%
  group_by(market_segment, is_canceled) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(market_segment) %>%
  mutate(percentage = count / sum(count)) %>%
  filter(is_canceled == 1) %>%
  ggplot(aes(x = reorder(market_segment, percentage), y = percentage)) +
  geom_bar(stat = "identity", fill = "coral") +
  labs(title = "Cancellation Rate by Market Segment", x = "Market Segment", y = "Cancellation Rate") +
  theme_minimal() +
  coord_flip()
print(p3)

p4 <- ggplot(train_data, aes(x = total_nights, fill = is_canceled)) +
  geom_histogram(binwidth = 1, position = "dodge") +
  labs(title = "Length of Stay vs Cancellation", x = "Total Nights", y = "Count") +
  theme_minimal() +
  scale_x_continuous(limits = c(0, 15))
print(p4)

numeric_vars <- train_data %>%
  select(where(is.numeric))

correlation_matrix <- cor(numeric_vars)
print(corrplot(correlation_matrix, method = "circle"))

logistic_model <- train_data %>%
  glm(formula = is_canceled ~ lead_time + arrival_date_month + 
        stays_in_weekend_nights + stays_in_week_nights + 
        market_segment + distribution_channel + is_repeated_guest + 
        previous_cancellations + booking_changes + 
        deposit_type + adr + total_of_special_requests,
      family = "binomial")

rf_model <- randomForest(is_canceled ~ ., data = train_data, ntree = 50, importance = TRUE)

train_data_xgb <- train_data
test_data_xgb <- test_data

convert_factors <- function(df) {
  df_processed <- df
  factor_columns <- sapply(df, is.factor)
  
  for (col in names(df)[factor_columns]) {
    if (col != "is_canceled") {
      df_processed[[col]] <- as.numeric(df[[col]])
    }
  }
  
  return(df_processed)
}

train_data_xgb <- convert_factors(train_data_xgb)
test_data_xgb <- convert_factors(test_data_xgb)

train_matrix <- as.matrix(train_data_xgb %>% select(-is_canceled))
train_label <- as.numeric(train_data_xgb$is_canceled) - 1

test_matrix <- as.matrix(test_data_xgb %>% select(-is_canceled))
test_label <- as.numeric(test_data_xgb$is_canceled) - 1

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgboost(
  data = train_matrix,
  label = train_label,
  params = xgb_params,
  nrounds = 50,
  verbose = 0
)

lr_pred_prob <- predict(logistic_model, newdata = test_data, type = "response")
lr_pred_class <- ifelse(lr_pred_prob > 0.5, 1, 0)
lr_pred_class <- factor(lr_pred_class, levels = c(0, 1))

rf_pred_prob <- predict(rf_model, newdata = test_data, type = "prob")[, 2]
rf_pred_class <- predict(rf_model, newdata = test_data)

xgb_pred_prob <- predict(xgb_model, newdata = test_matrix)
xgb_pred_class <- ifelse(xgb_pred_prob > 0.5, 1, 0)
xgb_pred_class <- factor(xgb_pred_class, levels = c(0, 1))

evaluate_model <- function(actual, predicted, predicted_prob, model_name) {
  cm <- confusionMatrix(predicted, actual)
  roc_obj <- roc(actual, predicted_prob)
  auc_value <- auc(roc_obj)
  
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  return(list(
    model = model_name,
    accuracy = cm$overall["Accuracy"],
    precision = precision,
    recall = recall,
    f1_score = f1,
    auc = auc_value,
    confusion_matrix = cm$table
  ))
}

lr_eval <- evaluate_model(test_data$is_canceled, lr_pred_class, lr_pred_prob, "Logistic Regression")
rf_eval <- evaluate_model(test_data$is_canceled, rf_pred_class, rf_pred_prob, "Random Forest")
xgb_eval <- evaluate_model(test_data$is_canceled, xgb_pred_class, xgb_pred_prob, "XGBoost")

evaluation_results <- data.frame(
  Model = c(lr_eval$model, rf_eval$model, xgb_eval$model),
  Accuracy = c(lr_eval$accuracy, rf_eval$accuracy, xgb_eval$accuracy),
  Precision = c(lr_eval$precision, rf_eval$precision, xgb_eval$precision),
  Recall = c(lr_eval$recall, rf_eval$recall, xgb_eval$recall),
  F1_Score = c(lr_eval$f1_score, rf_eval$f1_score, xgb_eval$f1_score),
  AUC = c(lr_eval$auc, rf_eval$auc, xgb_eval$auc)
)

rf_importance <- importance(rf_model)
rf_importance_df <- data.frame(
  Feature = rownames(rf_importance),
  Importance = rf_importance[, "MeanDecreaseGini"]
)
rf_importance_df <- rf_importance_df %>%
  arrange(desc(Importance)) %>%
  head(15)

xgb_importance <- xgb.importance(model = xgb_model)
xgb_importance_df <- data.frame(
  Feature = xgb_importance$Feature,
  Importance = xgb_importance$Gain
) %>%
  arrange(desc(Importance)) %>%
  head(15)

roc_lr <- roc(test_data$is_canceled, lr_pred_prob)
roc_rf <- roc(test_data$is_canceled, rf_pred_prob)
roc_xgb <- roc(test_data$is_canceled, xgb_pred_prob)

roc_plot <- ggroc(list(
  "Logistic Regression" = roc_lr,
  "Random Forest" = roc_rf,
  "XGBoost" = roc_xgb
)) +
  labs(
    title = "ROC Curves for Different Models",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  annotate("text", x = 0.75, y = 0.25,
           label = paste("AUC (LR):", round(lr_eval$auc, 3), "\n",
                         "AUC (RF):", round(rf_eval$auc, 3), "\n",
                         "AUC (XGB):", round(xgb_eval$auc, 3)))
print(roc_plot)

feature_importance_plot <- ggplot(rf_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Top 15 Features by Importance (Random Forest)", x = "Feature", y = "Importance") +
  theme_minimal() +
  coord_flip()
print(feature_importance_plot)

calibration_data <- data.frame(
  actual = as.numeric(test_data$is_canceled) - 1,
  predicted_prob = rf_pred_prob
)

calibration_data <- calibration_data %>%
  mutate(bin = cut(predicted_prob, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)) %>%
  group_by(bin) %>%
  summarise(
    mean_pred = mean(predicted_prob),
    mean_actual = mean(actual),
    count = n()
  )

calibration_plot <- ggplot(calibration_data, aes(x = mean_pred, y = mean_actual)) +
  geom_point(aes(size = count), color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  labs(title = "Calibration Plot for Random Forest Model",
       x = "Predicted Probability",
       y = "Actual Proportion") +
  theme_minimal() +
  xlim(0, 1) +
  ylim(0, 1)
print(calibration_plot)

cost_benefit <- data.frame(
  Prediction = c("True Negative", "False Positive", "False Negative", "True Positive"),
  Description = c(
    "Correctly predicted non-cancellation",
    "Incorrectly predicted cancellation",
    "Incorrectly predicted non-cancellation",
    "Correctly predicted cancellation"
  ),
  Business_Value = c(
    50,
    -10,
    -30,
    20
  )
)

cm <- confusionMatrix(rf_pred_class, test_data$is_canceled)
tn <- cm$table[1, 1]
fp <- cm$table[1, 2]
fn <- cm$table[2, 1]
tp <- cm$table[2, 2]

business_impact <- (tn * cost_benefit$Business_Value[1]) +
  (fp * cost_benefit$Business_Value[2]) +
  (fn * cost_benefit$Business_Value[3]) +
  (tp * cost_benefit$Business_Value[4])

business_impact_per_booking <- business_impact / nrow(test_data)

cat("\n--- Model Evaluation Results ---\n")
print(evaluation_results)

cat("\n--- Top 10 Important Features (Random Forest) ---\n")
print(head(rf_importance_df, 10))

cat("\n--- Business Impact Analysis ---\n")
cat("Total Business Value:", business_impact, "\n")
cat("Business Value per Booking:", business_impact_per_booking, "\n")
