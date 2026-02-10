"""
Configuration management for RKLLM converter.
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ConversionConfig:
    """Configuration for model conversion."""

    model_path: str
    lora_path: Optional[str] = None
    quantization: Literal["w8a8", "w4a16"] = "w8a8"
    context_length: int = 4096
    platform: Literal["RK3588", "RK3576", "RK3562", "RV1126B"] = "RK3588"
    npu_cores: int = 3
    output_path: Optional[str] = None
    dataset_path: Optional[str] = None
    quantization_algorithm: Optional[Literal["normal", "grq"]] = None
    dry_run: bool = False
    device: Literal["cpu", "cuda"] = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"

    def get_quantization_algorithm(self) -> str:
        """Get recommended quantization algorithm based on dtype."""
        if self.quantization_algorithm:
            return self.quantization_algorithm
        if self.quantization == "w4a16":
            return "grq"
        return "normal"

    def get_output_filename(self) -> str:
        """Generate output filename based on model and settings."""
        model_name = self.model_path.split("/")[-1]
        lora_suffix = ""
        if self.lora_path:
            lora_name = self.lora_path.split("/")[-1]
            lora_suffix = f"_{lora_name}"
        
        return f"{model_name}{lora_suffix}_{self.quantization}_ctx{self.context_length}_{self.platform}.rkllm"

    def get_optimization_level(self) -> int:
        """Get recommended optimization level."""
        return 0


DEFAULT_CALIBRATION_SAMPLES = [
    {
        "input": "在农业生产中被当作极其重要的劳动对象发挥作用，最主要的不可替代的基本生产资料是\nA. 农业生产工具\nB. 土地\nC. 劳动力\nD. 资金",
        "target": "B. 土地"
    },
    {
        "input": "Imagine you are participating in a race with a group of people. If you have just overtaken second person, what's your current position?",
        "target": "You are now in second position."
    },
    {
        "input": "def fizz_buzz(n: int): \"\"\"Return the number of times digit 7 appears in integers less than n which are divisible by 11 or 13.\"\"\"",
        "target": "def fizz_buzz(n):\n    count = 0\n    for i in range(n):\n        if i % 11 == 0 or i % 13 == 0:\n            count += str(i).count('7')\n    return count"
    },
    {
        "input": "Some people got on a bus at the terminal. At the first bus stop, half of the people got down and 4 more people got in. Then at the second stop, 6 people got down and 8 more got in. If there were 25 people heading to the third stop, how many people got on at the terminal?",
        "target": "Let x be the number who got on at the terminal. After first stop: x - x/2 + 4 = x/2 + 4. After second stop: (x/2 + 4) - 6 + 8 = x/2 + 6 = 25. So x/2 = 19, x = 38."
    },
    {
        "input": "把这句话翻译成英文: RK3588是新一代高端处理器，具有高算力、低功耗、超强多媒体、丰富数据接口等特点",
        "target": "RK3588 is a new generation of high-end processor, featuring high computing power, low power consumption, excellent multimedia performance, and rich data interfaces."
    },
    {
        "input": "在城市夜晚的霓虹灯下，车水马龙，您能为此创作七言绝句吗？关键词：夜晚，霓虹。",
        "target": "霓虹闪烁夜未央，车水马龙路正长。万家灯火映天际，都市风情醉心房。"
    },
    {
        "input": "What is the coefficient of $x^2y^6$ in the expansion of $(\\frac{3}{5}x-\\frac{y}{2})^8$?",
        "target": "Using binomial theorem: C(8,2) * (3/5)^2 * (-1/2)^6 = 28 * (9/25) * (1/64) = 252/1600 = 63/400"
    },
    {
        "input": "You are an AI assistant. Please help me understand how to use a list comprehension in Python.",
        "target": "List comprehensions in Python provide a concise way to create lists. Syntax: [expression for item in iterable if condition]. Example: [x**2 for x in range(10)] creates [0,1,4,9,16,25,36,49,64,81]."
    },
    {
        "input": "《坛经》，是历史上除佛经外，唯一被尊称为'经'的佛教典籍。此书作者是\nA. 六祖慧能\nB. 南天竺菩提达摩\nC. 释迦牟尼\nD. 五祖宏忍",
        "target": "A. 六祖慧能"
    },
    {
        "input": "下列行为如满足规定条件，应认定为伪造货币罪的是\nA. 将英镑揭层一分为二\nB. 铸造珍稀古钱币\nC. 临摹欧元收藏\nD. 用黄金铸造流通的纪念金币",
        "target": "A. 将英镑揭层一分为二"
    },
    {
        "input": "def is_multiply_prime(a): \"\"\"Write a function that returns true if given number is multiplication of 3 prime numbers and false otherwise. Example: is_multiply_prime(30) == True because 30 = 2 * 3 * 5\"\"\"",
        "target": "def is_multiply_prime(a):\n    primes = []\n    n = a\n    i = 2\n    while i * i <= n and len(primes) < 3:\n        if n % i == 0:\n            primes.append(i)\n            while n % i == 0:\n                n //= i\n        i += 1\n    if n > 1:\n        primes.append(n)\n    return len(primes) == 3 and a == eval('*'.join(map(str, primes)))"
    },
    {
        "input": "When long-run equilibrium is reached, a perfectly competitive firm always\nA. has economic profit greater than zero\nB. has normal profit equal to zero\nC. has economic profit less than zero\nD. has economic profit equal to zero",
        "target": "D. has economic profit equal to zero"
    },
    {
        "input": "Given the following Python code, rewrite it using list comprehension:\nsquares = []\nfor i in range(10):\n    squares.append(i**2)\nprint(squares)",
        "target": "squares = [i**2 for i in range(10)]\nprint(squares)"
    },
    {
        "input": "Knowledge can be acquired from many sources. These include books, teachers and practical experience, and each has its own advantages.",
        "target": "知识可以从许多来源获得。这些包括书籍、教师和实践经验，各有其优势。"
    },
    {
        "input": "设f(x)是偶函数，φ(x)是奇函数，则下列函数中是奇函数的是 ( )\nA. f[f(x)]\nB. φ[φ(x)]\nC. φ[f(x)]\nD. f[φ(x)]",
        "target": "D. f[φ(x)]"
    },
    {
        "input": "你是一个手机内负责日程管理的智能助手，你要基于用户给定的目标日程信息，综合考虑日程库的行程，联想一些可以体现人文关怀，实用，并且可以给用户带来惊喜的子日程提醒。",
        "target": "好的，我会根据您的主日程，智能推荐相关的子提醒，比如会议前提醒您准备材料，出差前提醒查看天气和交通，生日前提醒您准备礼物等，让您的日程安排更加贴心周到。"
    },
    {
        "input": "I love programming and I love learning new technologies. Help me improve my coding skills.",
        "target": "Here are some tips to improve your coding skills: 1. Practice daily with small projects. 2. Read and analyze others' code. 3. Learn design patterns. 4. Contribute to open source. 5. Stay updated with new technologies. 6. Get feedback through code reviews."
    },
    {
        "input": "RK3588 processor specifications include 8-core Cortex-A76/A55 CPU, ARM Mali-G610 GPU, and 6 TOPS NPU.",
        "target": "RK3588处理器规格包括8核Cortex-A76/A55 CPU、ARM Mali-G610 GPU和6 TOPS NPU。"
    },
    {
        "input": "Aussie Speedo Guy is a Bisexual Aussie Guy who loves speedos. He enjoys swimming at the beach and participating in water sports competitions.",
        "target": "澳大利亚泳裤男是一位喜欢泳裤的双性恋澳大利亚男子。他喜欢在海滩游泳并参加水上运动比赛。"
    },
    {
        "input": "What is theory of general relativity?",
        "target": "General relativity is a theory of gravitation developed by Albert Einstein. It describes gravity not as a force, but as a curvature of spacetime caused by mass and energy. The theory predicts phenomena like gravitational time dilation, light bending, and black holes."
    }
]
