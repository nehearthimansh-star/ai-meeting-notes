require("dotenv").config();

const express = require("express");
const multer = require("multer");
const axios = require("axios");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const { exec } = require("child_process");

const app = express();
app.use(cors());
app.use(express.static("public"));

const PORT = Number(process.env.PORT) || 3000;
const PYTHON_BIN = process.env.PYTHON_BIN || "python";
const uploadsDir = path.join(__dirname, "uploads");

fs.mkdirSync(uploadsDir, { recursive: true });

app.get("/health", (req, res) => {
    res.json({ ok: true });
});

const HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions";
const HF_CHAT_MODEL = "katanemo/Arch-Router-1.5B:hf-inference";
const HF_API_KEY = (
    process.env.HF_API_KEY ||
    process.env.HUGGINGFACE_API_KEY ||
    process.env.HF_TOKEN ||
    ""
).trim();
const STOP_WORDS = new Set([
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "because",
    "been", "before", "being", "but", "by", "can", "could", "did", "do", "does", "for", "from",
    "get", "got", "had", "has", "have", "he", "her", "here", "him", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "just", "let", "like", "may", "me", "more", "most", "my",
    "need", "now", "of", "on", "or", "our", "out", "please", "really", "said", "she", "should",
    "so", "some", "that", "the", "their", "them", "there", "they", "this", "to", "too", "up",
    "us", "was", "we", "were", "what", "when", "which", "who", "will", "with", "would", "you",
    "your"
]);
const TOPIC_LABELS = [
    { label: "attendance", pattern: /\b(attendance|present|absent|missing|late|student|students)\b/i },
    { label: "planning", pattern: /\b(plan|planning|schedule|timeline|calendar|next week|next month)\b/i },
    { label: "project progress", pattern: /\b(project|progress|status|milestone|delivery|launch)\b/i },
    { label: "team coordination", pattern: /\b(team|owner|owners|coordination|handoff|follow up|follow-up)\b/i },
    { label: "communication", pattern: /\b(email|message|communicat|inform|share|update|announce)\b/i },
    { label: "reporting", pattern: /\b(report|reports|data|dashboard|metrics|numbers)\b/i },
    { label: "budget", pattern: /\b(budget|cost|price|expense|spend|funding)\b/i },
    { label: "customer issues", pattern: /\b(customer|client|user|complaint|feedback|support)\b/i }
];
const CONCERN_LABELS = [
    { label: "attendance gaps", pattern: /\b(absent|missing|skipping|low attendance|late)\b/i },
    { label: "schedule pressure", pattern: /\b(delay|delayed|behind|deadline|timeline|urgent)\b/i },
    { label: "resource constraints", pattern: /\b(shortage|limited|resource|budget|cost)\b/i },
    { label: "communication gaps", pattern: /\b(confusion|unclear|not informed|miscommunication|follow up)\b/i },
    { label: "quality issues", pattern: /\b(issue|problem|error|bug|mistake|rework)\b/i }
];
const NOISE_PATTERNS = [
    /\bhello everyone\b/i,
    /\bhi everyone\b/i,
    /\bthank you guys? for coming\b/i,
    /\bthanks everyone\b/i,
    /\blet'?s just get started\b/i,
    /\byou know\b/i,
    /\bi think\b/i,
    /\bit might\b/i,
    /\bkind of\b/i,
    /\bsort of\b/i
];
const SIGNAL_PATTERNS = [
    /\b(attendance|student|students|school|project|team|plan|planning|schedule|timeline|issue|problem|risk|delay|follow up|follow-up|share|send|prepare|report|support|meeting|action|decision|update|poster|tips|sick|health)\b/i
];

function formatErrorMessage(value, fallback = "Unknown error") {
    if (!value) {
        return fallback;
    }

    if (typeof value === "string") {
        return value;
    }

    if (typeof value === "object") {
        if (typeof value.error === "string") {
            return value.error;
        }

        if (typeof value.message === "string") {
            return value.message;
        }

        try {
            return JSON.stringify(value);
        } catch (err) {
            return fallback;
        }
    }

    return String(value);
}

function getHfConfigError() {
    if (!HF_API_KEY) {
        return "Missing Hugging Face API key. Add HF_API_KEY to your .env file.";
    }

    if (!/^hf_/i.test(HF_API_KEY)) {
        return "Invalid Hugging Face API key format. Use a token that starts with hf_.";
    }

    return "";
}

function normalizeForComparison(text) {
    return (text || "")
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function isSummaryTooCloseToTranscript(summary, transcript) {
    const normalizedSummary = normalizeForComparison(summary);
    const normalizedTranscript = normalizeForComparison(transcript);

    if (!normalizedSummary) {
        return true;
    }

    if (normalizedTranscript.includes(normalizedSummary) && normalizedSummary.length > 40) {
        return true;
    }

    const summaryWords = normalizedSummary.split(" ").filter(Boolean);
    if (!summaryWords.length) {
        return true;
    }

    const transcriptWordSet = new Set(normalizedTranscript.split(" ").filter(Boolean));
    const overlapCount = summaryWords.filter((word) => transcriptWordSet.has(word)).length;
    const overlapRatio = overlapCount / summaryWords.length;

    return overlapRatio > 0.88;
}

function isLineTooCloseToTranscript(line, transcript, maxOverlap = 0.8) {
    const normalizedLine = normalizeForComparison(line);
    const normalizedTranscript = normalizeForComparison(transcript);

    if (!normalizedLine) {
        return true;
    }

    if (normalizedTranscript.includes(normalizedLine) && normalizedLine.length > 24) {
        return true;
    }

    const lineWords = normalizedLine.split(" ").filter(Boolean);
    if (!lineWords.length) {
        return true;
    }

    const transcriptWordSet = new Set(normalizedTranscript.split(" ").filter(Boolean));
    const overlapCount = lineWords.filter((word) => transcriptWordSet.has(word)).length;
    return overlapCount / lineWords.length > maxOverlap;
}

function splitSentences(text) {
    return text
        .replace(/\s+/g, " ")
        .split(/(?<=[.!?])\s+/)
        .map((sentence) => sentence.trim())
        .filter(Boolean);
}

function dedupeLines(lines) {
    const seen = new Set();
    return lines.filter((line) => {
        const key = line.toLowerCase();
        if (seen.has(key)) {
            return false;
        }
        seen.add(key);
        return true;
    });
}

function cleanFragment(text) {
    return text
        .replace(/^(hello everyone|hi everyone|thanks everyone|thank you guys for coming|let's just get started)[,.\s]*/i, "")
        .replace(/\b(i think|i've been noticing|i have|we have|you know|kind of|sort of)\b/gi, "")
        .replace(/\s+/g, " ")
        .trim()
        .replace(/^[,.\s]+/, "")
        .replace(/[.?\s]+$/, "");
}

function toSentence(text) {
    if (!text) {
        return "";
    }

    const trimmed = text.trim();
    return trimmed.charAt(0).toUpperCase() + trimmed.slice(1).replace(/[.?!\s]*$/, "") + ".";
}

function cleanListItem(text) {
    return text
        .replace(/^[-*]\s*/, "")
        .replace(/^(hello everyone|hi everyone|thanks everyone|thank you guys? for coming|let's just get started)[,.\s]*/i, "")
        .replace(/\b(let's|we should|i think|it might be good if|maybe|you know)\b/gi, "")
        .replace(/\s+/g, " ")
        .trim()
        .replace(/^[,.\s]+/, "")
        .replace(/[.?\s]+$/, "");
}

function contentWordCount(text) {
    return normalizeForComparison(text)
        .split(" ")
        .filter((word) => word.length > 2 && !STOP_WORDS.has(word))
        .length;
}

function isMeaningfulSentence(text) {
    const cleaned = cleanListItem(text);

    if (!cleaned || cleaned.length < 12) {
        return false;
    }

    if (NOISE_PATTERNS.some((pattern) => pattern.test(cleaned))) {
        return false;
    }

    if (contentWordCount(cleaned) < 3) {
        return false;
    }

    return SIGNAL_PATTERNS.some((pattern) => pattern.test(cleaned));
}

function filterMeaningfulLines(lines) {
    return dedupeLines((lines || []).map((line) => cleanListItem(line)).filter((line) => isMeaningfulSentence(line)));
}

function clipWords(text, maxWords = 12) {
    const words = cleanListItem(text).split(" ").filter(Boolean);
    return words.slice(0, maxWords).join(" ");
}

function extractFirstMatch(text, pattern, fallback = "") {
    const match = (text || "").match(pattern);
    return match ? match[1] || match[0] : fallback;
}

function extractQuestionsFromTranscript(transcript) {
    const questions = splitSentences(transcript).filter((sentence) => sentence.includes("?"));
    return questions.length ? questions.slice(0, 4) : ["No open questions."];
}

function extractActionItemsFromTranscript(transcript) {
    const actionItems = splitSentences(transcript).filter((sentence) =>
        isMeaningfulSentence(sentence) &&
        /\b(should|need to|needs to|plan to|let's|try|follow up|put up|send|share|prepare|remind|schedule)\b/i.test(sentence)
    );

    return dedupeLines(actionItems).slice(0, 5);
}

function extractKeyPointsFromTranscript(transcript) {
    const sentences = filterMeaningfulLines(splitSentences(transcript));
    const prioritized = sentences.filter((sentence) =>
        /\b(should|need|plan|decide|decision|problem|issue|important|next|action|question|because|let's)\b/i.test(sentence)
    );

    const fallback = prioritized.length ? prioritized : sentences;
    return dedupeLines(fallback).slice(0, 5);
}

function buildSummary(transcript, keyPoints, questions) {
    const normalizedTranscript = normalizeForComparison(transcript);
    const matchedTopics = TOPIC_LABELS
        .filter((topic) => topic.pattern.test(transcript))
        .map((topic) => topic.label);
    const matchedConcern = CONCERN_LABELS.find((item) => item.pattern.test(transcript))?.label || "";
    const hasActionItems = actionItemsLikelyPresent(keyPoints, transcript);
    const keywordTopics = extractTopKeywords(transcript);
    const topicPhrase = buildTopicPhrase(matchedTopics, keywordTopics);

    const parts = [];

    parts.push(topicPhrase
        ? `This meeting centered on ${topicPhrase}.`
        : "This meeting centered on current updates and immediate priorities.");

    if (matchedConcern) {
        parts.push(`A recurring concern was ${matchedConcern}, which affected the discussion.`);
    } else if (/\b(problem|issue|risk|concern|delay|missing|absent)\b/i.test(normalizedTranscript)) {
        parts.push("The team reviewed the main risks and blockers affecting progress.");
    } else {
        parts.push("The conversation focused on reviewing the present situation and areas needing attention.");
    }

    if (keyPoints.length) {
        parts.push("Participants aligned on the most important discussion points and clarified the overall direction.");
    } else {
        parts.push("Participants shared updates and aligned on the current direction of work.");
    }

    if (hasActionItems) {
        parts.push("Clear follow-up responsibilities and next steps were identified before closing the discussion.");
    } else {
        parts.push("The discussion concluded with a general understanding of what should happen next.");
    }

    if (questions.length && questions[0] !== "No open questions.") {
        parts.push("Some items remain open and will need additional follow-up after the meeting.");
    } else {
        parts.push("Overall, the meeting helped move the group toward a more coordinated plan.");
    }

    if (matchedTopics.length > 1 || keywordTopics.length > 1) {
        parts.push("The notes reflect both the broader context and the practical actions discussed by the group.");
    }

    return parts.slice(0, 6).join("\n").trim() || "Summary not available.";
}

function actionItemsLikelyPresent(keyPoints, transcript) {
    const combined = `${(keyPoints || []).join(" ")} ${transcript || ""}`;
    return /\b(should|need to|needs to|plan to|follow up|follow-up|share|send|prepare|schedule|remind|complete|submit|update)\b/i.test(combined);
}

function extractTopKeywords(text) {
    const words = normalizeForComparison(text)
        .split(" ")
        .filter((word) => word.length > 3 && !STOP_WORDS.has(word));
    const counts = new Map();

    words.forEach((word) => {
        counts.set(word, (counts.get(word) || 0) + 1);
    });

    return [...counts.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([word]) => word);
}

function buildTopicPhrase(matchedTopics, keywordTopics) {
    const topics = dedupeLines([
        ...matchedTopics,
        ...keywordTopics
    ]).slice(0, 3);

    if (!topics.length) {
        return "";
    }

    if (topics.length === 1) {
        return topics[0];
    }

    if (topics.length === 2) {
        return `${topics[0]} and ${topics[1]}`;
    }

    return `${topics[0]}, ${topics[1]}, and ${topics[2]}`;
}

function summarizeSentenceTopic(sentence) {
    const topic = extractFirstMatch(
        sentence,
        /\b(attendance|planning|schedule|timeline|project|progress|team|communication|report|budget|support|customer|delivery|quality)\b/i,
        ""
    );
    return topic ? topic.toLowerCase() : "";
}

function rewriteKeyPoint(sentence, transcript) {
    const cleaned = cleanListItem(sentence);

    if (!isMeaningfulSentence(cleaned)) {
        return "";
    }

    const topic = summarizeSentenceTopic(cleaned);

    if (/\b(decide|decision|agreed|finalized|confirmed)\b/i.test(cleaned)) {
        return `Decision aligned on ${topic || "the discussed priority"}.`;
    }

    if (/\b(problem|issue|risk|concern|delay|missing|absent|late)\b/i.test(cleaned)) {
        return `Risk noted around ${topic || "an active issue"}.`;
    }

    if (/\b(plan|planning|next|timeline|schedule)\b/i.test(cleaned)) {
        return `Next-step planning was discussed for ${topic || "upcoming work"}.`;
    }

    if (/\b(share|update|report|inform|communicat)\b/i.test(cleaned)) {
        return `An update was shared regarding ${topic || "team communication"}.`;
    }

    const clipped = clipWords(cleaned, 8).toLowerCase();
    const candidate = clipped ? `Discussion point related to ${clipped}.` : "";
    return candidate && !isLineTooCloseToTranscript(candidate, transcript, 0.9)
        ? candidate
        : `Discussion point related to ${topic || "current priorities"}.`;
}

function rewriteActionItem(sentence) {
    const cleaned = cleanListItem(sentence);

    if (!isMeaningfulSentence(cleaned)) {
        return "";
    }

    const owner = extractFirstMatch(cleaned, /\b([A-Z][a-z]+)\s+(?:will|should|needs to|need to)\b/, "");
    const objectPhrase = extractFirstMatch(
        cleaned,
        /\b(?:will|should|needs to|need to|plan to|let's|let us|try to|follow up on|prepare|share|send|schedule|remind)\b\s+(.+)/i,
        cleaned
    );
    const shortObject = clipWords(objectPhrase, 9).toLowerCase();

    if (/\b(schedule|arrange|plan)\b/i.test(cleaned)) {
        return `${owner || "Team"} to schedule the next follow-up.`;
    }

    if (/\b(send|share|inform|update)\b/i.test(cleaned)) {
        return `${owner || "Team"} to share the required update with stakeholders.`;
    }

    if (/\b(prepare|create|complete|submit)\b/i.test(cleaned)) {
        return `${owner || "Team"} to complete ${shortObject || "the pending deliverable"}.`;
    }

    if (/\b(remind|follow up|follow-up)\b/i.test(cleaned)) {
        return `${owner || "Team"} to follow up on outstanding items.`;
    }

    return `${owner || "Team"} to take the next step on ${shortObject || "the discussed work"}.`;
}

function sanitizeAiList(lines, transcript, rewriter, minimumCount, fallbackLines) {
    const cleaned = dedupeLines(
        filterMeaningfulLines(lines)
            .map((line) => (isLineTooCloseToTranscript(line, transcript) ? rewriter(line, transcript) : line))
            .map((line) => cleanListItem(line))
    ).filter(Boolean);

    if (cleaned.length >= minimumCount) {
        return cleaned.slice(0, 5);
    }

    return dedupeLines([...(cleaned || []), ...(fallbackLines || [])]).slice(0, 5);
}

function buildFallbackKeyPoints(transcript) {
    return extractKeyPointsFromTranscript(transcript)
        .map((sentence) => rewriteKeyPoint(sentence, transcript))
        .filter(Boolean)
        .slice(0, 5);
}

function buildFallbackActionItems(transcript) {
    const actions = extractActionItemsFromTranscript(transcript)
        .map((sentence) => rewriteActionItem(sentence))
        .filter(Boolean);

    return actions.length ? actions.slice(0, 5) : ["Team to confirm the next follow-up steps."];
}

function buildFallbackNotes(transcript) {
    const keyPoints = buildFallbackKeyPoints(transcript);
    const questions = extractQuestionsFromTranscript(transcript);
    const actionItems = buildFallbackActionItems(transcript);
    const summary = buildSummary(transcript, keyPoints, questions);

    return {
        summary: summary || "Summary not available.",
        keyPoints,
        questions,
        actionItems
    };
}

function parseSectionLines(text, heading) {
    const escapedHeading = heading.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pattern = new RegExp(`${escapedHeading}:\\s*([\\s\\S]*?)(?=\\n[A-Za-z ]+:|$)`, "i");
    const match = text.match(pattern);

    if (!match) {
        return [];
    }

    return match[1]
        .split("\n")
        .map((line) => line.replace(/^[-*]\s*/, "").trim())
        .filter(Boolean);
}

const storage = multer.diskStorage({
    destination: uploadsDir,
    filename: (req, file, cb) => {
        const ext = path.extname(file.originalname);
        cb(null, Date.now() + ext);
    }
});

const upload = multer({ storage });

app.post("/upload", upload.single("audio"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "Please select an audio file first." });
        }

        const filePath = req.file.path;
        console.log("Uploaded file:", filePath);

        exec(`"${PYTHON_BIN}" transcribe.py "${filePath}"`, async (err, stdout, stderr) => {
            console.log("STDOUT:", stdout);
            console.log("STDERR:", stderr);

            if (err) {
                console.error("Whisper error:", err);
                return res.status(500).json({ error: stderr || "Whisper error" });
            }

            const transcript = stdout.trim();

            if (!transcript) {
                return res.status(400).json({ error: "No transcript generated" });
            }

            console.log("Transcript:", transcript);

            const fallbackNotes = buildFallbackNotes(transcript);
            const hfConfigError = getHfConfigError();

            if (hfConfigError) {
                return res.json({
                    transcript,
                    summary: fallbackNotes.summary,
                    keyPoints: fallbackNotes.keyPoints,
                    actionItems: fallbackNotes.actionItems,
                    warning: hfConfigError
                });
            }

            try {
                const hfRes = await axios.post(
                    HF_CHAT_URL,
                    {
                        model: HF_CHAT_MODEL,
                        messages: [
                            {
                                role: "system",
                                content: "You are an AI assistant that creates professional meeting notes. Rewrite ideas into concise business language instead of copying transcript lines."
                            },
                            {
                                role: "user",
                                content: `Summarize the following meeting transcript in a CLEAR and SHORT way.

Rules:
- Do NOT copy sentences directly from the transcript
- Write the Summary as 5-6 short lines
- Each summary line should sound like meeting notes, not transcript text
- Paraphrase the discussion into a clean overview
- Rewrite Key Points and Action Items in polished note-style language
- Avoid fillers, repeated wording, and speaker-style phrasing
- Use simple professional language

Then also provide:

Summary:
- 5-6 short lines

Key Points:
- 4-5 clean bullets written as note-style takeaways

Action Items:
- 3-5 short actionable tasks in professional language

Transcript:
${transcript}`
                            }
                        ],
                        max_tokens: 350,
                        temperature: 0.2
                    },
                    {
                        headers: {
                            Authorization: `Bearer ${HF_API_KEY}`,
                            "Content-Type": "application/json"
                        }
                    }
                );

                const rawNotes = (
                    hfRes.data?.choices?.[0]?.message?.content ||
                    ""
                ).trim();

                const summaryLines = parseSectionLines(rawNotes, "Summary");
                const keyPoints = parseSectionLines(rawNotes, "Key Points");
                const actionItems = parseSectionLines(rawNotes, "Action Items");

                const aiSummary = summaryLines.join("\n").trim();
                const finalSummary =
                    aiSummary && !isSummaryTooCloseToTranscript(aiSummary, transcript)
                        ? aiSummary
                        : fallbackNotes.summary;
                const finalKeyPoints = sanitizeAiList(
                    keyPoints,
                    transcript,
                    rewriteKeyPoint,
                    3,
                    fallbackNotes.keyPoints
                );
                const finalActionItems = sanitizeAiList(
                    actionItems,
                    transcript,
                    rewriteActionItem,
                    2,
                    fallbackNotes.actionItems
                );

                res.json({
                    transcript,
                    summary: finalSummary,
                    keyPoints: finalKeyPoints,
                    actionItems: finalActionItems,
                    rawNotes
                });
            } catch (hfError) {
                const warningMessage = formatErrorMessage(
                    hfError.response?.data?.error ||
                    hfError.response?.data?.message ||
                    hfError.response?.data ||
                    hfError.message,
                    "AI notes unavailable, showing fallback notes."
                );
                const statusCode = hfError.response?.status;
                const authWarning = statusCode === 401 || statusCode === 403
                    ? "Hugging Face authentication failed. Check the HF_API_KEY value in your .env file."
                    : warningMessage;

                console.error("HF ERROR:", authWarning);
                res.json({
                    transcript,
                    summary: fallbackNotes.summary,
                    keyPoints: fallbackNotes.keyPoints,
                    actionItems: fallbackNotes.actionItems,
                    warning: authWarning
                });
            }
        });
    } catch (err) {
        console.error("SERVER ERROR:", err);
        res.status(500).json({ error: err.message || "Server error" });
    }
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
