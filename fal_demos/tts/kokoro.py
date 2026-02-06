from typing import Literal

import fal
from fal.exceptions import FieldException
from fal.toolkit import File
from fastapi import Response
from pydantic import BaseModel, Field


class AmEnglishRequest(BaseModel):
    prompt: str = Field(
        default="",
        examples=[
            "The future belongs to those who believe in the beauty of their dreams. So, dream big, work hard, and make it happen!"
        ],
        ui={"important": True},
    )
    text: str = Field(
        default="",
        examples=[
            "The future belongs to those who believe in the beauty of their dreams. So, dream big, work hard, and make it happen!"
        ],
    )
    # Use Literal for voice to restrict to specific enum values
    voice: Literal[
        "af_heart",
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
    ] = Field(
        examples=["af_heart"],
        default="af_heart",
        description="Voice ID for the desired voice.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Speed of the generated audio. Default is 1.0.",
    )


class BrEnglishRequest(BaseModel):
    prompt: str = Field(
        examples=[
            "Ladies and gentlemen, welcome aboard. Please ensure your seatbelt is fastened and your tray table is stowed as we prepare for takeoff."
        ]
    )
    voice: Literal[
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
    ] = Field(
        examples=["bf_alice"],
        description="Voice ID for the desired voice.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Speed of the generated audio. Default is 1.0.",
    )


class JapaneseRequest(BaseModel):
    prompt: str = Field(
        examples=[
            "夢を追いかけることを恐れないでください。努力すれば、必ず道は開けます！"
        ]
    )
    voice: Literal[
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
    ] = Field(
        examples=["jf_alpha"],
        description="Voice ID for the desired voice.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Speed of the generated audio. Default is 1.0.",
    )


class AmEngOutput(BaseModel):
    audio: File = Field(
        description="The generated music",
        examples=[
            File._from_url(
                "https://fal.media/files/elephant/dXVMqWsBDG9yan3kaOT0Z_tmp0vvkha3s.wav"
            )
        ],
    )


class BrEngOutput(BaseModel):
    audio: File = Field(
        description="The generated music",
        examples=[
            File._from_url(
                "https://fal.media/files/kangaroo/4wpA60Kum6UjOVBKJoNyL_tmpxfrkn95k.wav"
            )
        ],
    )


class JapaneseOutput(BaseModel):
    audio: File = Field(
        description="The generated music",
        examples=[
            File._from_url(
                "https://fal.media/files/lion/piLhqKO8LJxrWaNg2dVUv_tmpp6eff6zl.wav"
            )
        ],
    )


class Kokoro(
    fal.App,
    min_concurrency=0,  # type: ignore
    max_concurrency=1,  # type: ignore
    keep_alive=3000,  # type: ignore
    name="kokoro",  # type: ignore
):
    requirements = [
        "kokoro==0.8.4",
        "soundfile==0.13.1",
        "misaki[en]==0.8.4",
        "misaki[ja]==0.8.4",
        "misaki[zh]==0.8.4",
        "numpy==1.26.4",
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
    ]
    machine_type = "L"  # Use a CPU machine type since Kokoro is only 82M parameters and runs efficiently on CPU

    async def setup(self):
        from kokoro import KPipeline

        self.pipelines = {}
        self.pipelines["American English"] = KPipeline(lang_code="a")
        self.pipelines["British English"] = KPipeline(lang_code="b")
        self.pipelines["Japanese"] = KPipeline(lang_code="j")

    async def _generate(
        self,
        request: AmEnglishRequest,
        response: Response,
        language: str = "American English",
    ):
        prompt = request.prompt or request.text
        if len(prompt) >= 20000:
            # Use fieldexception to nicely render the error in the UI
            raise FieldException(
                field="prompt",
                message="Prompt must be less than 20000 characters.",
            )

        import tempfile

        import numpy as np
        import soundfile as sf

        pipeline = self.pipelines[language]
        generator = pipeline(
            prompt,
            voice=request.voice,
            speed=request.speed,
            split_pattern=r"\n+",
        )
        for i, (gs, ps, audio) in enumerate(generator):
            if i == 0:
                final_audio = audio.detach().cpu().numpy()
            else:
                audio = audio.detach().cpu().numpy()
                final_audio = np.concatenate((final_audio, audio), axis=0)

        # Set the billing units to be a minimum of 1 and scale with 1000 characters
        response.headers["x-fal-billable-units"] = str(max(1, len(prompt) // 1000))

        # Use a temporary file to save the audio and then send it via the cdn through the File object
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, final_audio, 24000)
            return AmEngOutput(
                audio=File.from_path(f.name, content_type="audio/wav", repository="cdn")
            )

    @fal.endpoint("/")
    async def generate(
        self, request: AmEnglishRequest, response: Response
    ) -> AmEngOutput:
        return await self._generate(request, response, language="American English")

    @fal.endpoint("/american-english")
    async def generate_am_english(
        self, request: AmEnglishRequest, response: Response
    ) -> AmEngOutput:
        return await self._generate(request, response, language="American English")

    @fal.endpoint("/british-english")
    async def generate_br_english(
        self, request: BrEnglishRequest, response: Response
    ) -> BrEngOutput:
        return await self._generate(request, response, language="British English")

    @fal.endpoint("/japanese")
    async def generate_japanese(
        self, request: JapaneseRequest, response: Response
    ) -> JapaneseOutput:
        return await self._generate(request, response, language="Japanese")


...  # Define the rest of the languages as endpoints similarly
