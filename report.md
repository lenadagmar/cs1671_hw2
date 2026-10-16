## 1. Pre-processing Steps

To clean the text, I lowercased it, replaced all non-word characters (punctuation and symbols) with spaces, and stripped extra whitespace. I removed stopwords. I used only functions (lower(), sub(), strip()) from Python's reguar expressions module.


## 2. Performance table

Evaluated on `diplomacy_dev.csv`. Precision, recall, and F1 are for the positive (lying) class.

| Model | Features | # Features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 1 (baseline) | Unigram counts | 2649 | 0.640 | 0.687 | 0.517 | 0.590 |
| 2 | Unigram counts + `winning?` | 2650 | 0.640 | 0.687 | 0.517 | 0.590 |
| 3 | Binary unigrams + `winning?` | 2650 | 0.680 | 0.716 | 0.596 | 0.650 |

## 3. Feature descriptions

**Modification 1: `winning?` (from `meta.game_score_delta`)**

1. *Motivation:* Deception in Diplomacy may depend on whether or not a player is winning or losing. I originally hypothesized that maybe, if a player was winning, they would be less likely to lie. If they have the advantage already, why be deceptive? So, using the meta.game_score_delta column, I used the score delta to create the 'winning?' flag: 1 if player is winning (their score delta is >0), and 0 if player is not winning (score delta <=0. 
2. *Results:* Adding `winning?` did not change any dev metric: Model 2 matches the baseline on accuracy (0.640), precision (0.687), recall (0.517), and F1 (0.590). However, it did not mean what I thought it would. The learned coefficient was +0.32, so the model associated winning/being in the lead with a **higher** chance of a deceptive message. This association, although not strong, was interesting. I suppose if someone is winning, it may be because they are better at being deceptive, and more likely to employ that tactic.

**Modification 2: binary unigrams**

1. *Motivation:* Messages are short, so a word repeated several times is probably not much more informative than the word appearing once. Binary features reduce the influence of long or repetitive messages and may reduce overfitting.
2. *Results:* Switching from counts to binary presence gave a small improvement over Model 2: accuracy 0.640 to 0.680, precision 0.687 to 0.716, recall 0.517 to 0.596, and F1 0.590 to 0.650. 

## 4. Informative features and error analysis

Model used: Model 3 (binary unigrams + `winning?`)

### 4.1 Informative features

| Rank | Most positive (deception) | Weight | Rank | Most negative (truthful) | Weight |
|---|---|---|---|---|---|
| 1 | alright | 0.771 | 1 | convoying | -0.614 |
| 2 | stop | 0.723 | 2 | doesn | -0.581 |
| 3 | real | 0.699 | 3 | makes | -0.566 |
| 4 | trying | 0.678 | 4 | winningflag | -0.550 |
| 5 | okay | 0.672 | 5 | player | -0.470 |

- **"alright"** (weight 0.771)
  - Appears in 22 training messages, 81.8% labeled lies
  - Examples in context:
    1. [LIE] "Alright, just didn't want to overstep the lines, you know how it is. I can probably even use it to keep up the subterfuge even with supporting you…"
    2. [truth] "Alright, they probably spoke to you in the same window of time that they did to me. Did you talk about which of you gets Belgium?"
    3. [LIE] "Alright excellent. Sounds good. We'll even it out. Promise that. I just want a healthy border around my home centers and a guarantee I can push back France"

- **"stop"** (weight 0.723)
  - Appears in 35 training messages, 65.7% labeled lies
  - Examples in context:
    1. [LIE] "Ugh idk. I might have to remain neutral there another season, I can't afford England coming after me from the north or east (and he's agreed to stop at Moscow for now)…"
    2. [truth] "I am starting to get frustrated with England, who has not replied to my messages for over a year while clowning around instead of helping stop the Italy Austria alliance…"
    3. [LIE] "I think the point where I stop making any progress is coming up very soon, so I'd love to start talking about ways this plays out to a draw"

- **"real"** (weight 0.699)
  - Appears in 6 training messages, 100.0% labeled lies
  - Examples in context:
    1. [LIE] "Oh sorry. I've been real busy"
    2. [LIE] "Hang on this is good: we're gonna see what his real priorities are! He wanted me to cut support from Boh, and I refused."
    3. [LIE] "I'm so confused -- it's not like you have any real chance of eliminating me at this point"

- **"trying"** (weight 0.678)
  - Appears in 37 training messages, 67.6% labeled lies
  - Examples in context:
    1. [truth] "Potentially, but I'm worried that you and France are trying to push me out of this triple"
    2. [LIE] "Re G being mad: it's not the end of the world… I just don't want her trying to throw the game in spite or anything like that. Cause I am lying to her like crazy this turn."
    3. [truth] "What I was/am trying to avoid is a scenario in which you (a) work with Germany and (b) send way more strength to the Med than you can actually use…"

- **"okay"** (weight 0.672)
  - Appears in 23 training messages, 82.6% labeled lies
  - Examples in context:
    1. [LIE] "Okay it's all good"
    2. [LIE] "So just to be clear, I'll be moving into Pie and staying there, then convoying tus-tun this turn. Then I'll move wes-mao. That all sound okay?"
    3. [LIE] "Okay. If he does say anything to you about moving towards me, would you please let me know?…"

- **"convoying"** (weight -0.614)
  - Appears in 5 training messages, 20.0% labeled lies
  - Examples in context:
    1. [truth] "...okay, clearly I don't have your trust. But really, I am committed to an FG victory! I'm convoying bel onto britain this spring and I'll leave it open for you…"
    2. [truth] "I'll try but honestly, probably can't- He's not going to be happy about my fleet build. Right now we're talking about convoying Tunis into Greece which, lol"
    3. [LIE] "So just to be clear, I'll be moving into Pie and staying there, then convoying tus-tun this turn…"

- **"doesn"** (weight -0.581, from "doesn't")
  - Appears in 14 training messages, 50.0% labeled lies
  - Examples in context:
    1. [LIE] "Hey Germany, Everybody vastly overreacted… Even though I clearly wasn't going to…"
    2. [LIE] "Well... that was harsh. Let me just put a few things straight. Me and Italy is not a thing…"
    3. [truth] "So I just realized that my mind made a goof and Warsaw doesn't border Bohemia. Would you mind terribly if I moved into Galicia…"

- **"makes"** (weight -0.566)
  - Appears in 34 training messages, 44.1% labeled lies
  - Examples in context:
    1. [truth] "Right makes sense"
    2. [LIE] "Italy getting two builds concerns me a little bit… losing you would be crippling for my game."
    3. [LIE] "I mean, you will definitely be in the draw if that's what you meant. I would have 100% proposed this beforehand…"

- **`winningflag`** (the `winning?` indicator, weight -0.550)
  - Lie rate for winning players: 58.3% (n = 557); for non-winning players: 44.6% (n = 861).
  
- **"player"** (weight -0.470)
  - Appears in 8 training messages, 62.5% labeled lies
  - Examples in context:
    1. [LIE] "I spoke to England and they don't seem like a particularly diplomatic player, but I'm pretty sure I can get into the English channel…"
    2. [LIE] "Well... that was harsh. Let me just put a few things straight…"
    3. [truth] "Yeah. I may have to ally with england to stop Austria. Apparently, he is the top ranked player in the world."

I think many of these examples are just noise and don't provide much insight into the data. One thing that stuck out to me were the words "okay" and "alright" indicating deception. I think maybe words of reassurance could be used subconciously by someone who is lying to make them seem more credible. It's also curious to me that winning was an indicator of truthfullness more than deception in this model. It had a coefficient of +.32 in the first model I used it in. I think it may have to do with the standardization step.

### 4.2 Confusion matrix and error analysis

|  | Predicted truthful | Predicted lying |
|---|---|---|
| **Actual truthful** | [68] | [21] |
| **Actual lying** | [36] | [53] |

**False negatives (lies predicted truthful), 36 total:**
1. "If you think I should then I will"
2. "I inquired about Italy a little bit but not so much about other plans"
3. "I can offer keeping you in the game so you make the draw."

**False positives (truthful predicted lying), 21 total:**
1. "Alright, will do my best!"
2. "That's true. I'm trying to keep it that way (no enemies)."
3. "BTW I'm feeding Austria a heaping helping of bullshit right now to try and get them to back off of me… I'm still playing them"

**Discussion:**
- *Patterns:* The false negatives are short, plain, and contain none of the top deceptive words, so the model defaults to "truthful". The false positives include top deceptive words: "alright" and "trying".
- *Possible fixes:* The model sees each message alone so a useful addition would be the earlier messages, to discern whether a given player has a pattern of being deceptive. this combined with text analysis and logisitc regression would significantly improve the prediction.
