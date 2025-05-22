from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_name = "mamba-2.8b-hf"

# It's good practice to set the pad_token if it's None,
# though for this model, it's already set to eos_token_id.
tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/{model_name}")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token # Common practice
# print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}, eos_token_id: {tokenizer.eos_token_id}")


model = MambaForCausalLM.from_pretrained(f"state-spaces/{model_name}").to(device)
model.eval() # Set to evaluation mode if not training

# --- Your original input ---
prompt = """
what is the root cause of the SWERR (software error) given in this chunk of data.
Give this data as context:

"Oct 30 00:35:38 fujitsu esalbase[458]: GetDuplex:101

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 no duplex

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 no default vlan

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 unknown port access

Oct 30 00:35:38 fujitsu esalbase[458]: SettingNotFoundException:MAIN.portType.OSC1.memberVlansnot found

Oct 30 00:35:38 fujitsu esalbase[458]: portType=OSC1 No member vlans

Oct 30 00:35:38 fujitsu esalbase[458]: portType=OSC1 No memberVlanRangeStart

Oct 30 00:35:38 fujitsu esalbase[458]: portType=OSC1 No memberVlanRangeEnd

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 no nniMode

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 dtagMode=DTAG_MODE_EXTERNAL

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 in vlan translation valid=1

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 inVlanBase=3950

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 inVlanOffset=0

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 inVlanBase=3900

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 inVlanTransOffset=0

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 in vlan translation valid=1

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 inVlanBase=2050

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 inVlanOffset=0

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 inVlanBase=2060

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 inVlanTransOffset=0

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 out vlan translation valid=1

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 outVlanBase=3900

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 outVlanOffset=0

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 outVlanBase=3950

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 outVlanTransOffset=0

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 out vlan translation valid=1

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 outVlanBase=2060

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 outVlanOffset=0

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 outVlanBase=2050

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC1 outVlanTransOffset=0

Oct 30 00:35:38 fujitsu esalbase[458]: SettingNotFoundException:MAIN.portType.OSC1.portAdvertCapabilitynot found

Oct 30 00:35:38 fujitsu esalbase[458]: portType=OSC1 No portAdvertCap

Oct 30 00:35:38 fujitsu esalbase[458]: portType'OSC1portAdvertCapability=63

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC2 no rate

Oct 30 00:35:38 fujitsu esalbase[458]: GetDuplex:101

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC2 no duplex

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC2 defaultVlan=4000

Oct 30 00:35:38 fujitsu esalbase[458]:  portType=OSC2 unknown port access

Oct 30 00:35:38 fujitsu esalbase[458]: SettingNotFoundException:MAIN.portType.OSC2.memberVlansnot found

Oct 30 00:35:38 fujitsu systemd[1]: Starting Pkt Handler App startup service file...

Oct 30 00:35:38 fujitsu FNC_FACILITY=SWERR[458]: =====================SWERR Start=====================

                                                 Sequence Number: 50

                                                 Level: KS_SWERR_ONLY

                                                 TID:   esalbase(0)

                                                 PID:   esalbase(0x1ca)

                                                 PPID:  systemd(0x1)

                                                 Core:  0

                                                 File:  /usr/src/debug/esal-base/1.9+javelin+gitr67+1da78a8c37-r67/git/src/esalBoardFramework.cc:609

                                                 Stack Trace:

                                                 /usr/lib/libswerr.so.0(_ZN12SwerrContext10init_earlyEv+0x90)[0x14b0f4eb8550]

                                                 /usr/lib/libswerr.so.0(_ZN12SwerrContext4initEv+0x9)[0x14b0f4ebbbf9]

                                                 /usr/lib/libswerr.so.0(_ZN5SwerrC1ENS_10SwerrLevelERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEiS8_PK12SwerrContext+0x16f)[0x14b0f4ebbd7f]

                                                 /usr/bin/esalbase(_ZN18EsalBoardFramework11IsPortValidEj+0x1e3)[0x556f1d73ab73]

                                                 /usr/bin/esalbase(_ZN18EsalBoardFramework14HandlePortProvE9InterfaceRK15PortProvMessage+0x3c4)[0x556f1d74e3d4]

                                                 /usr/bin/esalbase(_ZN13EsalStaticCfg14ConfigurePortsEv+0x9b)[0x556f1d6e977b]

                                                 /usr/bin/esalbase(_ZN18EsalBoardFramework9BoardInitEv+0x45)[0x556f1d73a355]

                                                 /usr/bin/esalbase(_Z12esalBaseMainv+0x11c)[0x556f1d6cc39c]

                                                 /usr/bin/esalbase(main+0x9)[0x556f1d6c7129]

                                                 /lib/libc.so.6(+0x2d57b)[0x14b0f451657b]

                                                 /lib/libc.so.6(__libc_start_main+0x80)[0x14b0f4516630]

                                                 /usr/bin/esalbase(_start+0x25)[0x556f1d6c7185]

                                                 Application Info:

                                                 Invalid Port, port=25

                                                 ======================SWERR End======================"
 

"""

# --- Tokenize and ensure attention_mask is included ---
# The tokenizer will return input_ids and attention_mask by default
# when return_tensors="pt"
inputs = tokenizer(prompt, return_tensors="pt")

# Move inputs to the same device as the model
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device) # Crucial for the warning

# print(f"Input IDs shape: {input_ids.shape}")
# print(f"Attention Mask shape: {attention_mask.shape}")
# print(f"Input IDs: {input_ids}")
# print(f"Attention Mask: {attention_mask}")


# --- Generate output, providing the attention_mask ---
# For Mamba, max_length is often preferred over max_new_tokens with generate,
# but max_new_tokens should also work.
# Let's set eos_token_id explicitly for generation, and pad_token_id
# (though for single unpadded input, pad_token_id's role in generation is minimal)
with torch.no_grad(): # Important for inference
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask, # Pass the attention mask
        max_new_tokens=100,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, # Good practice
        do_sample=True,
        top_k=50,
        temperature=0.7,
    )

# Decode the output
# `out` will contain the input_ids as well, so if you only want the generated part:
generated_ids = out[:, input_ids.shape[1]:]
print("prompt: ", prompt)
print("response:", tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

# print(tokenizer.batch_decode(out, skip_special_tokens=True))