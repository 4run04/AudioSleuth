param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
Add-Type -ReferencedAssemblies System.Speech -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Speech.Synthesis;
using System.Speech.AudioFormat;

public class Boundary {
    public string Text { get; set; }
    public double Start { get; set; }
}
public static class DemoVoice {
    public static Boundary[] Render(string text, string voice, int rate, string path) {
        var boundaries = new List<Boundary>();
        using (var synth = new SpeechSynthesizer()) {
            synth.SelectVoice(voice);
            synth.Rate = rate;
            synth.Volume = 85;
            synth.SetOutputToWaveFile(path,
                new SpeechAudioFormatInfo(16000, AudioBitsPerSample.Sixteen, AudioChannel.Mono));
            synth.SpeakProgress += delegate(object sender, SpeakProgressEventArgs e) {
                int end = e.CharacterPosition + e.CharacterCount;
                while (end < text.Length && ".,!?;:".IndexOf(text[end]) >= 0) end++;
                boundaries.Add(new Boundary {
                    Text = text.Substring(e.CharacterPosition, end - e.CharacterPosition),
                    Start = e.AudioPosition.TotalSeconds
                });
            };
            synth.Speak(text);
        }
        return boundaries.ToArray();
    }
}
'@

$assets = Join-Path $ProjectRoot 'assets'
$output = Join-Path $assets '_generated'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$turns = Get-Content -LiteralPath (Join-Path $assets 'script.json') -Raw | ConvertFrom-Json
$records = @()
for ($i = 0; $i -lt $turns.Count; $i++) {
    $turn = $turns[$i]
    $path = Join-Path $output ("turn-{0:D2}.wav" -f $i)
    $events = [DemoVoice]::Render($turn.text, $turn.voice, $turn.rate, $path)
    $records += [pscustomobject]@{
        speaker = $turn.speaker
        gain = $turn.gain
        audio = [System.IO.Path]::GetFileName($path)
        words = @($events)
    }
}
$records | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'events.json') -Encoding UTF8
Write-Output "Rendered $($records.Count) original synthetic turns in $output"
