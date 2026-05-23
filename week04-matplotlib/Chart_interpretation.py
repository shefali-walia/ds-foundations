# ==================================================================
# CHART INTERPRETATION — how to read charts like a data scientist
# ==================================================================
# The goal of any chart is to answer ONE of these questions:
#   1. How is this distributed?        (shape of data)
#   2. How does this change over time? (trend)
#   3. How do these relate?            (relationship/correlation)
#   4. How do categories compare?      (comparison)
#   5. What part makes up the whole?   (composition)
#
# For every chart you look at, ask in order:
#   WHAT   -- what variables are on each axis, what are the units
#   SHAPE  -- what is the overall pattern
#   TREND  -- is there a direction (up/down/flat/cyclical)
#   SPREAD -- how much variation exists (tight cluster vs wide scatter)
#   OUTLIERS -- any points/bars far from the rest
#   SO WHAT -- what does this mean for the real-world question
# ==================================================================


# ------------------------------------------------------------------
# 1. LINE CHART (single line)
# ------------------------------------------------------------------
# USE WHEN: one numerical variable over time (or ordered sequence)
#
# WHAT TO LOOK FOR:
#   overall direction  -- upward trend, downward trend, flat, U-shape
#   seasonality        -- repeating pattern at regular intervals
#   volatility         -- how jagged/smooth the line is
#   breakpoints        -- sudden level shifts (event happened here)
#   rate of change     -- steep slope = fast change, gentle = slow
#
# QUESTIONS TO ASK:
#   is the trend consistent or does it change at some point?
#   are the dips/peaks random or do they repeat at the same time each year?
#   is the line getting more or less volatile over time?
#
# EXAMPLE LANGUAGE:
#   "shows an upward trend with strong yearly seasonality"
#   "volatile in the short term but stable long-term direction"
#   "sharp drop in [period] suggests an external disruption"
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 2. MULTIPLE LINE CHART
# ------------------------------------------------------------------
# USE WHEN: comparing trends of 2+ groups over time
#
# WHAT TO LOOK FOR:
#   gap between lines   -- which group is consistently higher/lower
#   gap changing        -- are lines converging or diverging over time
#   same pattern        -- do all lines move together (common driver)
#   different pattern   -- lines diverge at some point (different drivers)
#   crossover points    -- one line overtakes another (shift in dominance)
#
# QUESTIONS TO ASK:
#   do the groups follow the same seasonal pattern?
#   is the gap between them growing, shrinking, or stable?
#   does one group react more strongly to the same conditions?
#
# EXAMPLE LANGUAGE:
#   "registered users consistently outpace casual users throughout"
#   "both lines follow the same seasonal cycle, suggesting a shared driver"
#   "gap widens in year 2, indicating growth is registered-led"
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 3. HISTOGRAM
# ------------------------------------------------------------------
# USE WHEN: understanding the distribution of a single numerical variable
#
# WHAT TO LOOK FOR:
#   shape:
#     symmetric/bell    -- normal distribution, mean ~ median
#     right skewed      -- long tail on right, most values are low
#                          mean > median (pulled by high outliers)
#     left skewed       -- long tail on left, most values are high
#                          mean < median
#     bimodal           -- two peaks = two distinct groups in your data
#     uniform           -- flat = roughly equal frequency across range
#   center              -- where is the peak (most common value range)
#   spread              -- narrow = low variance, wide = high variance
#   outliers            -- isolated bars far from the main body
#
# QUESTIONS TO ASK:
#   is this symmetric or skewed? which direction?
#   is there one peak or more than one?
#   are there gaps (missing values in a range)?
#   what does the shape tell us about the real-world variable?
#
# EXAMPLE LANGUAGE:
#   "right-skewed distribution -- most days have low casual rentals
#    but a few days see very high counts, pulling the mean up"
#   "bimodal shape suggests two distinct user behaviour patterns"
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 4. BOXPLOT
# ------------------------------------------------------------------
# USE WHEN: comparing distributions across categories, or seeing spread
#
# ANATOMY:
#   middle line  = median (Q2) -- the "typical" value
#   box edges    = Q1 (25th percentile) and Q3 (75th percentile)
#   box width    = IQR (interquartile range) = middle 50% of data
#   whiskers     = extend to 1.5 * IQR beyond Q1/Q3
#   dots outside = outliers (beyond whisker range)
#
# WHAT TO LOOK FOR:
#   median position     -- high or low relative to axis range
#   box width (IQR)     -- wide = high variability, narrow = consistent
#   whisker length      -- long = data spreads far, short = tight range
#   outlier dots        -- how many, how extreme, which direction
#   median position inside box:
#     centered          -- roughly symmetric distribution
#     closer to Q1      -- right skewed
#     closer to Q3      -- left skewed
#
# WHEN COMPARING MULTIPLE BOXPLOTS (e.g. one per season):
#   which category has the highest median
#   which has the most/least variability (IQR width)
#   do any categories have more outliers than others
#   do the boxes overlap or are they clearly separated
#
# EXAMPLE LANGUAGE:
#   "summer shows the highest median and widest IQR -- peak season
#    with the most variable demand"
#   "winter has a low median and several low-end outliers"
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 5. SCATTER PLOT
# ------------------------------------------------------------------
# USE WHEN: exploring relationship between two numerical variables
#
# WHAT TO LOOK FOR:
#   direction:
#     positive  -- points go up-right (both increase together)
#     negative  -- points go down-right (one up, other down)
#     none      -- no pattern (variables are unrelated)
#   strength:
#     strong    -- tight cluster around an imaginary line
#     moderate  -- visible trend but lots of spread
#     weak      -- barely any pattern, mostly scattered
#   shape:
#     linear    -- straight line pattern
#     curved    -- relationship exists but not linear (e.g. quadratic)
#     fan-shaped -- spread increases as x increases (heteroscedasticity)
#   clusters    -- separate groups of points (hidden categories)
#   outliers    -- points far from the main body
#
# QUESTIONS TO ASK:
#   if I drew a line through this, would it be straight or curved?
#   does the spread stay constant or widen as x increases?
#   are there isolated clusters that might represent subgroups?
#
# EXAMPLE LANGUAGE:
#   "moderate positive linear relationship -- as temp increases,
#    rentals generally increase, but with substantial spread"
#   "fan-shaped scatter suggests variance in y increases with x"
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 6. SCATTER PLOT WITH HUE (colour-encoded third variable)
# ------------------------------------------------------------------
# USE WHEN: scatter plot but you want to show a categorical third variable
#
# WHAT TO LOOK FOR:
#   do the colour groups cluster separately or overlap
#   does the relationship (direction/strength) differ by colour group
#   is one colour group consistently higher/lower in y
#   are there colours that appear only in certain x ranges
#
# QUESTIONS TO ASK:
#   does adding colour reveal a pattern that was hidden before?
#   does the trend hold for ALL colour groups or just some?
#
# EXAMPLE LANGUAGE:
#   "working days (orange) cluster at higher rental counts across
#    all temperatures compared to non-working days (blue)"
#   "the positive temp-rental relationship holds in both seasons
#    but is steeper in summer"
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 7. BUBBLE CHART
# ------------------------------------------------------------------
# USE WHEN: scatter plot + a third numerical variable encoded as bubble size
# effectively: 3 numerical variables in one chart
#
# WHAT TO LOOK FOR:
#   same as scatter for x vs y relationship
#   bubble size pattern -- do large bubbles cluster in a region?
#   correlation between size and position -- e.g. big bubbles = high x AND high y?
#   outlier bubbles     -- very large or very small in unexpected locations
#
# QUESTIONS TO ASK:
#   does the third variable (size) correlate with x, y, or both?
#   are the biggest bubbles where you'd expect them to be?
#
# EXAMPLE LANGUAGE:
#   "largest bubbles (highest humidity) cluster at moderate temperatures,
#    suggesting humidity peaks in mild weather"
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 8. BAR CHART (single)
# ------------------------------------------------------------------
# USE WHEN: comparing a numerical value across categories
#
# WHAT TO LOOK FOR:
#   tallest/shortest bar  -- best/worst performing category
#   relative differences  -- is one bar 2x another or barely different?
#   ordering             -- sorted bars reveal ranking clearly
#   unexpected values    -- a category you'd expect to be high is low
#
# QUESTIONS TO ASK:
#   how large is the difference between the highest and lowest bar?
#   is the pattern what you'd expect from domain knowledge?
#   would the insight change if y-axis started at 0 vs a higher value?
#   (always check if y-axis starts at 0 -- truncated axes exaggerate differences)
#
# EXAMPLE LANGUAGE:
#   "season 3 (fall) shows the highest average rentals, nearly double season 1"
#   "minimal difference between weekday categories suggests
#    day of week has limited impact on demand"
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 9. GROUPED BAR CHART
# ------------------------------------------------------------------
# USE WHEN: comparing multiple groups across multiple categories
# e.g. registered vs casual rentals, broken down by season
#
# WHAT TO LOOK FOR:
#   within each group   -- which bar is taller (which category wins)
#   across groups       -- does the pattern stay the same in every group
#   gap between bars    -- consistent gap = consistent difference
#   interaction effect  -- gap changes across groups = the two variables interact
#
# QUESTIONS TO ASK:
#   does one colour always dominate regardless of category?
#   is there any category where the usual pattern reverses?
#
# EXAMPLE LANGUAGE:
#   "registered users outnumber casual users in every season,
#    but the gap is smallest in summer -- casual usage peaks relatively more"
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 10. STACKED BAR CHART
# ------------------------------------------------------------------
# USE WHEN: showing both total size AND composition at once
# each bar = total, segments = parts of that total
#
# WHAT TO LOOK FOR:
#   total bar height     -- which category has the highest total
#   segment proportions  -- does the composition change across categories
#   dominant segment     -- which part is always the largest share
#   composition shift    -- a segment grows/shrinks as a proportion
#
# WATCH OUT:
#   hard to compare middle segments (no common baseline)
#   better for 2-3 segments max, more than that gets hard to read
#
# EXAMPLE LANGUAGE:
#   "registered users make up ~75% of total rentals in every season"
#   "casual share increases slightly in summer, visible in the wider
#    bottom segment"
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 11. SUBPLOTS (multiple charts in one figure)
# ------------------------------------------------------------------
# USE WHEN: comparing different variables or views side by side
#
# WHAT TO LOOK FOR:
#   shared patterns     -- do charts in the same row/col move together
#   contrast            -- where two subplots tell opposite stories
#   scale differences   -- different y-axis ranges can mislead if not noticed
#
# ALWAYS CHECK:
#   are y-axes on the same scale? if not, direct visual comparison is invalid
#   what is the intended comparison -- row vs row, or col vs col?
#
# EXAMPLE LANGUAGE:
#   "both subplots show the same seasonal cycle, confirming
#    that the pattern is present regardless of user type"
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 12. HEATMAP (correlation matrix)
# ------------------------------------------------------------------
# USE WHEN: seeing relationships between MANY numerical variables at once
#
# WHAT TO LOOK FOR:
#   dark cells (strong colour)  -- strong correlation (+ or -)
#   light/white cells           -- weak or no correlation
#   diagonal                    -- always = 1.0 (variable vs itself)
#   symmetric pattern           -- top-right mirrors bottom-left
#   clusters of correlated vars -- groups of variables that move together
#
# CORRELATION VALUES:
#   0.7 to 1.0   -- strong positive
#   0.4 to 0.7   -- moderate positive
#   0.0 to 0.4   -- weak positive
#   negative     -- as one increases, other decreases
#   close to 0   -- no linear relationship
#
# WATCH OUT:
#   correlation != causation
#   only captures LINEAR relationships (misses curved ones)
#   multicollinearity -- two predictors highly correlated with each other
#                        is a problem in ML models
#
# EXAMPLE LANGUAGE:
#   "temp and cnt show strong positive correlation (0.63)"
#   "casual and registered are both strongly correlated with cnt,
#    as expected since cnt = casual + registered"
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# UNIVERSAL RED FLAGS -- things that should always make you pause
# ------------------------------------------------------------------
#
# truncated y-axis    -- starts at non-zero, exaggerates differences
# overplotting        -- points stacked on top of each other hiding density
#                        fix: use alpha, or switch to hexbin/density plot
# missing axis labels -- you can't interpret a chart without knowing units
# too many categories -- more than 6-7 colours/groups = unreadable
# dual y-axes         -- can be manipulated to imply false relationships
# correlation claimed -- always ask: could a third variable explain both?
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# THE EXPERT WORKFLOW -- what to do when you see a new chart
# ------------------------------------------------------------------
#
# STEP 1: read the title and axis labels FIRST (before looking at data)
# STEP 2: note the scale (what are the min/max values, what are units)
# STEP 3: identify the chart type (what kind of question is it answering)
# STEP 4: find the most prominent pattern (what jumps out first)
# STEP 5: look for exceptions to that pattern (outliers, anomalies)
# STEP 6: connect back to the real-world question
#         "what does this mean for the problem we're trying to solve?"
# STEP 7: ask what this chart CANNOT tell you
#         (what questions remain unanswered, what would you plot next)
# ------------------------------------------------------------------