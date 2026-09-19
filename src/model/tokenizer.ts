export const tokenize = (text: string) => text.toLowerCase().match(/[a-z]+|[.!?,]/g) ?? [];
