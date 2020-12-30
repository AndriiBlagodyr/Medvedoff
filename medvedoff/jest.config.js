const path = require('path');

module.exports = {
	rootDir: path.join(__dirname, '.'),
	moduleDirectories: ['node_modules', path.join(__dirname, 'src'), 'public', __dirname],
	collectCoverage: true,
	collectCoverageFrom: ['src/**/*.{js,jsx,ts,tsx}', '!src/**/index.js'],
	coverageReporters: ['text-summary', 'json', 'html', 'lcovonly'],
	coverageThreshold: {
		global: {
			branches: 100,
			functions: 100,
			statements: 100,
			lines: 100
		}
	}
};
