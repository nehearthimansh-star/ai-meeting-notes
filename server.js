require("dotenv").config();

const express = require("express");
const multer = require("multer");
const axios = require("axios");
const cors = require("cors");
const path = require("path");
const { exec } = require("child_process");

const app = express();
app.use(cors());
app.use(express.static("public"));

const HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions";
const HF_CHAT_MODEL = "katanemo/Arch-Router-1.5B:hf-inference";

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
        .replace(/\b(let's|we should|i think|it might be good if|maybe|you know)\b/gi, "")
        .replace(/\s+/g, " ")
        .trim()
        .replace(/^[,.\s]+/, "")
        .replace(/[.?\s]+$/, "");
}

function extractQuestionsFromTranscript(transcript) {
    const questions = splitSentences(transcript).filter((sentence) => sentence.includes("?"));
    return questions.length ? questions.slice(0, 4) : ["No open questions."];
}

function extractActionItemsFromTranscript(transcript) {
    const actionItems = splitSentences(transcript).filter((sentence) =>
        /\b(should|need to|needs to|plan to|let's|try|follow up|put up|send|share|prepare|remind|schedule)\b/i.test(sentence)
    );

    return dedupeLines(actionItems).slice(0, 5);
}

function extractKeyPointsFromTranscript(transcript) {
    const sentences = splitSentences(transcript);
    const prioritized = sentences.filter((sentence) =>
        /\b(should|need|plan|decide|decision|problem|issue|important|next|action|question|because|let's)\b/i.test(sentence)
    );

    const fallback = prioritized.length ? prioritized : sentences;
    return dedupeLines(fallback).slice(0, 5);
}

function buildSummary(transcript, keyPoints, questions) {
    const sentences = splitSentences(transcript);
    const introSource = cleanFragment(sentences[0] || "");
    const issueSource = cleanListItem(
        sentences.find((sentence) => /\b(problem|issue|trend|concern|difficult|hard|absent|missing|skipping|sick)\b/i.test(sentence)) || ""
    );
    const actionSource = cleanListItem(
        keyPoints.find((point) => /\b(should|need|plan|try|put up|encourage|reminder|next week|support)\b/i.test(point)) || ""
    );
    const topicSource = cleanListItem(
        keyPoints.find((point) => /\b(meeting|students|attendance|school|project|team|report|planning)\b/i.test(point)) || introSource
    );

    const parts = [];

    if (topicSource) {
        parts.push(toSentence(`This meeting reviewed ${topicSource.toLowerCase()} and highlighted the main areas that need attention`));
    }

    if (issueSource) {
        parts.push(toSentence(`A key concern was ${issueSource.toLowerCase()}`));
    }

    if (actionSource) {
        parts.push(toSentence(`The discussion led to proposed next steps such as ${actionSource.toLowerCase()}`));
    }

    if (questions.length && questions[0] !== "No open questions.") {
        parts.push(toSentence("Some discussion points remain open and may require follow-up after the meeting"));
    } else if (keyPoints.length) {
        parts.push(toSentence("Overall, the conversation centered on practical actions and near-term follow-up"));
    }

    return parts.filter(Boolean).slice(0, 4).join(" ") || "Summary not available.";
}

function buildFallbackNotes(transcript) {
    const keyPoints = extractKeyPointsFromTranscript(transcript);
    const questions = extractQuestionsFromTranscript(transcript);
    const actionItems = extractActionItemsFromTranscript(transcript);
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
    destination: "uploads/",
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

        exec(`python transcribe.py "${filePath}"`, async (err, stdout, stderr) => {
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

            try {
                const hfRes = await axios.post(
                    HF_CHAT_URL,
                    {
                        model: HF_CHAT_MODEL,
                        messages: [
                            {
                                role: "system",
                                content: "You are an AI assistant that creates professional meeting notes."
                            },
                            {
                                role: "user",
                                content: `Summarize the following meeting transcript in a CLEAR and SHORT way.

Rules:
- Do NOT copy sentences directly
- Keep it concise (max 5-6 lines)
- Use simple professional language

Then also provide:

Summary:
- short summary

Key Points:
- bullet points

Action Items:
- short actionable tasks

Transcript:
${transcript}`
                            }
                        ],
                        max_tokens: 350,
                        temperature: 0.2
                    },
                    {
                        headers: {
                            Authorization: `Bearer ${process.env.HF_API_KEY}`,
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

                const aiSummary = summaryLines.join(" ").trim();
                const finalSummary =
                    aiSummary && !isSummaryTooCloseToTranscript(aiSummary, transcript)
                        ? aiSummary
                        : fallbackNotes.summary;

                res.json({
                    transcript,
                    summary: finalSummary,
                    keyPoints: keyPoints.length ? keyPoints : fallbackNotes.keyPoints,
                    actionItems: actionItems.length ? actionItems : fallbackNotes.actionItems,
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

                console.error("HF ERROR:", warningMessage);
                res.json({
                    transcript,
                    summary: fallbackNotes.summary,
                    keyPoints: fallbackNotes.keyPoints,
                    actionItems: fallbackNotes.actionItems,
                    warning: warningMessage
                });
            }
        });
    } catch (err) {
        console.error("SERVER ERROR:", err);
        res.status(500).json({ error: err.message || "Server error" });
    }
});

app.listen(3000, () => {
    console.log("Server running on http://localhost:3000");
});
