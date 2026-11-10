# Generates mock speech samples with the built-in Windows text-to-speech engine (no downloads),
# so the pipeline can be demonstrated end-to-end on real spoken language.
#   samples/meeting.wav   ~30 s, short file (sync path)
#   samples/long_talk.wav ~3 min, used with small chunks to demonstrate long-file handling
param([string]$OutDir = "$PSScriptRoot\..\samples")

Add-Type -AssemblyName System.Speech
New-Item -ItemType Directory -Force $OutDir | Out-Null

$meeting = @(
  "Good morning everyone, and thanks for joining the quarterly product review.",
  "Today we will cover three topics: customer feedback, the release schedule, and hiring.",
  "First, customer feedback. Support tickets dropped by eighteen percent after the March update.",
  "Second, the release schedule. Version two point four is planned for the fifteenth of July.",
  "Finally, hiring. We are looking for two backend engineers with experience in Python and cloud services.",
  "Please send any questions to the team before Friday. Thank you."
)

$talk = @(
  "Welcome to this short talk about building reliable data pipelines.",
  "A pipeline takes raw input, validates it, transforms it, and delivers it to the systems that need it.",
  "The first principle is to validate early. Reject bad input at the edge, before it wastes compute.",
  "The second principle is to normalize. Convert every input into one internal representation, so the rest of the system has a single code path.",
  "The third principle is to make work resumable. Large jobs should be split into pieces that can be retried independently.",
  "The fourth principle is observability. Record how long each stage takes, how much data it processed, and why it failed.",
  "Let us talk about audio specifically. Audio arrives in many containers and codecs, such as wave, M P three, M four A, and ogg.",
  "Sample rates vary from eight kilohertz telephone audio to forty eight kilohertz studio recordings.",
  "Speech recognition models usually expect sixteen kilohertz mono audio, so resampling is a required step.",
  "Long recordings, such as meetings or podcasts, can last for hours. Loading them entirely into memory is wasteful.",
  "Instead, we stream the audio, cut it into chunks at natural pauses, and transcribe each chunk.",
  "Each chunk's timestamps are then shifted by the chunk's offset, so the final transcript lines up with the original file.",
  "Cutting at pauses matters, because cutting in the middle of a word produces errors at every boundary.",
  "For production, long files should be processed asynchronously. The client receives a job identifier and polls for the result.",
  "Finally, the output should be useful downstream. Timestamps per segment allow subtitles, search, and linking an answer back to the exact moment it was said.",
  "Thank you for listening."
)

function Save-Speech([string[]]$lines, [string]$path, [int]$pauseMs) {
  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
  $s.Rate = 0
  $s.SetOutputToWaveFile($path)
  foreach ($line in $lines) { $s.Speak($line); $pb = New-Object System.Speech.Synthesis.PromptBuilder; $pb.AppendBreak([TimeSpan]::FromMilliseconds($pauseMs)); $s.Speak($pb) }
  $s.Dispose()
  Write-Host "wrote $path"
}

Save-Speech $meeting (Join-Path $OutDir "meeting.wav") 600
Save-Speech ($talk + $talk[0..5]) (Join-Path $OutDir "long_talk.wav") 700
